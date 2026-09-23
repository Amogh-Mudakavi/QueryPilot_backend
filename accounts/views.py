import logging
from datetime import timedelta

from django.conf import settings
from django.contrib.auth.hashers import make_password
from django.db import IntegrityError, transaction
from django.db.models import F
from django.utils import timezone
from rest_framework import status
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from utils.auth_utils import (
    decode_token,
    generate_access_token,
    generate_refresh_token,
    hash_refresh_token,
    refresh_token_expires_at,
    login_rate_limited,
    clear_login_rate_limits,
)
from utils.email_utils import (
    can_send_otp,
    check_hourly_limit,
    generate_otp,
    hash_otp,
    send_otp_email,
    verify_otp,
)

from .models import EmailOTP, PendingRegistration, RefreshToken, User
from .serializers import (
    EmailSerializer,
    LoginSerializer,
    RefreshSerializer,
    RegisterRequestSerializer,
    ResetPasswordSerializer,
    VerifyOTPSerializer,
)


logger = logging.getLogger(__name__)


def _client_ip(request):
    # REMOTE_ADDR is used intentionally. Do not trust a client-supplied
    # X-Forwarded-For value for abuse controls until the proxy is trusted.
    return request.META.get("REMOTE_ADDR")


def _invalid_otp_response():
    return Response(
        {"error": "Invalid or expired OTP."},
        status=status.HTTP_400_BAD_REQUEST,
    )


class RequestRegisterView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = RegisterRequestSerializer(data=request.data)

        if not serializer.is_valid():
            return Response(
                {"error": serializer.errors},
                status=status.HTTP_400_BAD_REQUEST,
            )

        email = serializer.validated_data["email"]
        ip_address = _client_ip(request)

        if not can_send_otp(email, ip_address):
            return Response(
                {
                    "error": "Please wait before requesting another OTP."
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        if not check_hourly_limit(email, ip_address):
            return Response(
                {
                    "error": (
                        "Maximum OTP limit reached. "
                        "Please try again later."
                    )
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        otp = generate_otp()

        hashed_password = make_password(
            serializer.validated_data["password"]
        )

        hashed_otp = hash_otp(otp)

        expires_at = timezone.now() + timedelta(
            minutes=settings.OTP_LIFETIME_MINUTES
        )

        try:
            with transaction.atomic():
                PendingRegistration.objects.filter(
                    email=email
                ).delete()

                EmailOTP.objects.filter(
                    email=email,
                    purpose="register",
                    is_used=False,
                ).delete()

                PendingRegistration.objects.create(
                    username=serializer.validated_data["username"],
                    email=email,
                    password=hashed_password,
                    expires_at=expires_at,
                )

                EmailOTP.objects.create(
                    email=email,
                    otp=hashed_otp,
                    purpose="register",
                    expires_at=expires_at,
                    last_sent_ip=ip_address,
                )

                send_otp_email(email, otp)

            return Response(
                {
                    "message": (
                        "OTP sent successfully. Please check your email "
                        "inbox and spam folder too."
                    )
                },
                status=status.HTTP_200_OK,
            )

        except IntegrityError:
            return Response(
                {
                    "error": (
                        "Could not process registration. "
                        "Please try again."
                    )
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        except Exception:
            logger.exception("Registration OTP flow failed")

            return Response(
                {
                    "error": (
                        "Could not process registration. "
                        "Please try again."
                    )
                },
                status=status.HTTP_400_BAD_REQUEST,
            )


class VerifyOTP(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = VerifyOTPSerializer(data=request.data)

        if not serializer.is_valid():
            return Response(
                {"error": serializer.errors},
                status=status.HTTP_400_BAD_REQUEST,
            )

        email = serializer.validated_data["email"]
        otp = serializer.validated_data["otp"]

        try:
            with transaction.atomic():
                otp_record = (
                    EmailOTP.objects
                    .select_for_update()
                    .filter(
                        email=email,
                        purpose="register",
                        is_used=False,
                    )
                    .first()
                )

                if otp_record is None:
                    return _invalid_otp_response()

                if otp_record.expires_at < timezone.now():
                    otp_record.is_used = True
                    otp_record.save(
                        update_fields=["is_used"]
                    )

                    PendingRegistration.objects.filter(
                        email=email
                    ).delete()

                    return _invalid_otp_response()

                if (
                    otp_record.verification_attempts
                    >= settings.OTP_MAX_VERIFICATION_ATTEMPTS
                ):
                    return Response(
                        {
                            "error": (
                                "Maximum verification attempts "
                                "reached."
                            )
                        },
                        status=status.HTTP_403_FORBIDDEN,
                    )

                if not verify_otp(
                    otp,
                    otp_record.otp,
                ):
                    EmailOTP.objects.filter(
                        pk=otp_record.pk
                    ).update(
                        verification_attempts=F(
                            "verification_attempts"
                        ) + 1
                    )

                    otp_record.refresh_from_db()

                    remaining = max(
                        settings.OTP_MAX_VERIFICATION_ATTEMPTS
                        - otp_record.verification_attempts,
                        0,
                    )

                    return Response(
                        {
                            "error": "Invalid OTP.",
                            "remaining_attempts": remaining,
                        },
                        status=status.HTTP_400_BAD_REQUEST,
                    )

                pending_user = (
                    PendingRegistration.objects
                    .select_for_update()
                    .filter(email=email)
                    .first()
                )

                if (
                    pending_user is None
                    or pending_user.expires_at < timezone.now()
                ):
                    otp_record.is_used = True
                    otp_record.save(
                        update_fields=["is_used"]
                    )

                    PendingRegistration.objects.filter(
                        email=email
                    ).delete()

                    return _invalid_otp_response()

                User.objects.create(
                    username=pending_user.username,
                    email=pending_user.email,
                    password=pending_user.password,
                )

                otp_record.is_used = True
                otp_record.save(
                    update_fields=["is_used"]
                )

                pending_user.delete()

            return Response(
                {
                    "message": "Account created successfully."
                },
                status=status.HTTP_201_CREATED,
            )

        except IntegrityError:
            return Response(
                {
                    "error": (
                        "Could not create account. "
                        "Please try again."
                    )
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        except Exception:
            logger.exception("Account creation failed")

            return Response(
                {
                    "error": (
                        "Could not create account. "
                        "Please try again."
                    )
                },
                status=status.HTTP_400_BAD_REQUEST,
            )


class ForgotPasswordRequestView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = EmailSerializer(data=request.data)

        if not serializer.is_valid():
            return Response(
                {"error": serializer.errors},
                status=status.HTTP_400_BAD_REQUEST,
            )

        email = serializer.validated_data["email"]
        ip_address = _client_ip(request)

        generic_response = Response(
            {
                "message": (
                    "If an account exists for this email, a password "
                    "reset OTP has been sent."
                )
            },
            status=status.HTTP_200_OK,
        )

        if not User.objects.filter(
            email=email
        ).exists():
            return generic_response

        if not can_send_otp(
            email,
            ip_address,
        ):
            return generic_response

        if not check_hourly_limit(
            email,
            ip_address,
        ):
            return generic_response

        otp = generate_otp()

        hashed_otp = hash_otp(otp)

        expires_at = timezone.now() + timedelta(
            minutes=settings.OTP_LIFETIME_MINUTES
        )

        try:
            with transaction.atomic():
                EmailOTP.objects.filter(
                    is_used=False,
                    purpose="reset_password",
                    email=email,
                ).delete()

                EmailOTP.objects.create(
                    email=email,
                    otp=hashed_otp,
                    purpose="reset_password",
                    expires_at=expires_at,
                    last_sent_ip=ip_address,
                )

                send_otp_email(email, otp)

        except Exception:
            logger.exception(
                "Password reset OTP flow failed"
            )

        return generic_response


class ResetPasswordView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = ResetPasswordSerializer(
            data=request.data
        )

        if not serializer.is_valid():
            return Response(
                {"error": serializer.errors},
                status=status.HTTP_400_BAD_REQUEST,
            )

        email = serializer.validated_data["email"]
        otp = serializer.validated_data["otp"]
        new_password = serializer.validated_data["new_password"]

        try:
            with transaction.atomic():
                otp_record = (
                    EmailOTP.objects
                    .select_for_update()
                    .filter(
                        email=email,
                        purpose="reset_password",
                        is_used=False,
                    )
                    .first()
                )

                if otp_record is None:
                    return _invalid_otp_response()

                if otp_record.expires_at < timezone.now():
                    otp_record.is_used = True
                    otp_record.save(
                        update_fields=["is_used"]
                    )

                    return _invalid_otp_response()

                if (
                    otp_record.verification_attempts
                    >= settings.OTP_MAX_VERIFICATION_ATTEMPTS
                ):
                    return Response(
                        {
                            "error": (
                                "Maximum verification attempts "
                                "exceeded."
                            )
                        },
                        status=status.HTTP_400_BAD_REQUEST,
                    )

                if not verify_otp(
                    otp,
                    otp_record.otp,
                ):
                    EmailOTP.objects.filter(
                        pk=otp_record.pk
                    ).update(
                        verification_attempts=F(
                            "verification_attempts"
                        ) + 1
                    )

                    otp_record.refresh_from_db()

                    remaining = max(
                        settings.OTP_MAX_VERIFICATION_ATTEMPTS
                        - otp_record.verification_attempts,
                        0,
                    )

                    return Response(
                        {
                            "error": "Invalid OTP.",
                            "remaining_attempts": remaining,
                        },
                        status=status.HTTP_400_BAD_REQUEST,
                    )

                user = (
                    User.objects
                    .select_for_update()
                    .filter(email=email)
                    .first()
                )

                if user is None:
                    otp_record.is_used = True
                    otp_record.save(
                        update_fields=["is_used"]
                    )

                    return _invalid_otp_response()

                user.set_password(new_password)

                user.save(
                    update_fields=["password"]
                )

                otp_record.is_used = True
                otp_record.save(
                    update_fields=["is_used"]
                )

                RefreshToken.objects.filter(
                    user=user,
                    is_revoked=False,
                ).update(
                    is_revoked=True
                )

            return Response(
                {
                    "message": "Password reset successfully."
                },
                status=status.HTTP_200_OK,
            )

        except Exception:
            logger.exception(
                "Password reset failed"
            )

            return Response(
                {
                    "error": (
                        "Could not reset password. "
                        "Please try again."
                    )
                },
                status=status.HTTP_400_BAD_REQUEST,
            )


class LoginView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        raw_email = request.data.get(
            "email",
            ""
        )

        email = (
            raw_email.strip().lower()
            if isinstance(raw_email, str)
            else ""
        )

        ip_address = _client_ip(request)

        if email and login_rate_limited(
            email,
            ip_address,
        ):
            return Response(
                {
                    "error": (
                        "Too many login attempts. "
                        "Please try again later."
                    )
                },
                status=status.HTTP_429_TOO_MANY_REQUESTS,
            )

        serializer = LoginSerializer(
            data=request.data
        )

        if not serializer.is_valid():
            return Response(
                {
                    "error": "Invalid credentials."
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        user = serializer.validated_data["user"]

        clear_login_rate_limits(
            email,
            ip_address,
        )

        access_token = generate_access_token(
            user
        )

        refresh_token = generate_refresh_token(
            user
        )

        RefreshToken.objects.create(
            user=user,
            token=hash_refresh_token(
                refresh_token
            ),
            expires_at=refresh_token_expires_at(),
        )

        return Response(
            {
                "access_token": access_token,
                "refresh_token": refresh_token,
            },
            status=status.HTTP_200_OK,
        )


class getme(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user

        return Response(
            {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "role": user.role,
            },
            status=status.HTTP_200_OK,
        )


class RefreshView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = RefreshSerializer(
            data=request.data
        )

        if not serializer.is_valid():
            return Response(
                serializer.errors,
                status=status.HTTP_400_BAD_REQUEST,
            )

        raw_refresh_token = (
            serializer.validated_data["refresh_token"]
        )

        try:
            payload = decode_token(
                raw_refresh_token
            )

        except Exception:
            return Response(
                {
                    "error": (
                        "Invalid or expired refresh token."
                    )
                },
                status=status.HTTP_401_UNAUTHORIZED,
            )

        if payload.get("type") != "refresh":
            return Response(
                {
                    "error": "Invalid token type."
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        token_hash = hash_refresh_token(
            raw_refresh_token
        )

        with transaction.atomic():
            try:
                token_obj = (
                    RefreshToken.objects
                    .select_for_update()
                    .select_related("user")
                    .get(
                        token=token_hash
                    )
                )

            except RefreshToken.DoesNotExist:
                return Response(
                    {
                        "error": "Invalid refresh token."
                    },
                    status=status.HTTP_401_UNAUTHORIZED,
                )

            if token_obj.user_id != payload.get(
                "user_id"
            ):
                return Response(
                    {
                        "error": "Invalid refresh token."
                    },
                    status=status.HTTP_401_UNAUTHORIZED,
                )

            if token_obj.is_revoked:
                RefreshToken.objects.filter(
                    user_id=token_obj.user_id,
                    is_revoked=False,
                ).update(
                    is_revoked=True
                )

                return Response(
                    {
                        "error": "Invalid refresh token."
                    },
                    status=status.HTTP_401_UNAUTHORIZED,
                )

            if token_obj.expires_at < timezone.now():
                token_obj.is_revoked = True

                token_obj.save(
                    update_fields=["is_revoked"]
                )

                return Response(
                    {
                        "error": (
                            "Refresh token expired. "
                            "Please login again."
                        )
                    },
                    status=status.HTTP_401_UNAUTHORIZED,
                )

            user = token_obj.user

            if not user.is_active or user.is_blocked:
                token_obj.is_revoked = True

                token_obj.save(
                    update_fields=["is_revoked"]
                )

                return Response(
                    {
                        "error": "Invalid refresh token."
                    },
                    status=status.HTTP_401_UNAUTHORIZED,
                )

            token_obj.is_revoked = True

            token_obj.save(
                update_fields=["is_revoked"]
            )

            new_refresh_token = (
                generate_refresh_token(user)
            )

            RefreshToken.objects.create(
                user=user,
                token=hash_refresh_token(
                    new_refresh_token
                ),
                expires_at=refresh_token_expires_at(),
            )

        access_token = generate_access_token(
            user
        )

        return Response(
            {
                "access_token": access_token,
                "refresh_token": new_refresh_token,
            },
            status=status.HTTP_200_OK,
        )


class LogoutView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = RefreshSerializer(
            data=request.data
        )

        if not serializer.is_valid():
            return Response(
                serializer.errors,
                status=status.HTTP_400_BAD_REQUEST,
            )

        raw_refresh_token = (
            serializer.validated_data["refresh_token"]
        )

        try:
            payload = decode_token(
                raw_refresh_token
            )

        except Exception:
            return Response(
                {
                    "error": "Refresh token not found."
                },
                status=status.HTTP_404_NOT_FOUND,
            )

        if payload.get("type") != "refresh":
            return Response(
                {
                    "error": "Refresh token not found."
                },
                status=status.HTTP_404_NOT_FOUND,
            )

        token_hash = hash_refresh_token(
            raw_refresh_token
        )

        with transaction.atomic():
            try:
                token_obj = (
                    RefreshToken.objects
                    .select_for_update()
                    .get(
                        token=token_hash
                    )
                )

            except RefreshToken.DoesNotExist:
                return Response(
                    {
                        "error": "Refresh token not found."
                    },
                    status=status.HTTP_404_NOT_FOUND,
                )

            if (
                token_obj.user_id != request.user.id
                or token_obj.user_id != payload.get("user_id")
            ):
                return Response(
                    {
                        "error": "Refresh token not found."
                    },
                    status=status.HTTP_404_NOT_FOUND,
                )

            if not token_obj.is_revoked:
                token_obj.is_revoked = True

                token_obj.save(
                    update_fields=["is_revoked"]
                )

        return Response(
            {
                "message": "Logged out successfully."
            },
            status=status.HTTP_200_OK,
        )