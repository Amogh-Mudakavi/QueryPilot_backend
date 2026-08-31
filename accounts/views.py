import logging
from datetime import timedelta

from django.conf import settings
from django.contrib.auth.hashers import make_password
from django.db import transaction
from django.db.models import F
from django.utils import timezone
from rest_framework import status
from rest_framework.response import Response
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.views import APIView

from utils.auth_utils import (
    decode_token,
    generate_access_token,
    generate_refresh_token,
)
from utils.email_utils import (
    can_send_otp,
    check_hourly_limit,
    generate_otp,
    send_otp_email,
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

        if not can_send_otp(email):
            return Response(
                {
                    "error": "Please wait 60 seconds before requesting another OTP."
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        if not check_hourly_limit(email):
            return Response(
                {"error": "Maximum OTP limit reached. Please try again later."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        otp = generate_otp()

        hashed_password = make_password(serializer.validated_data["password"])

        try:
            with transaction.atomic():

                PendingRegistration.objects.filter(email=email).delete()

                EmailOTP.objects.filter(
                    email=email, purpose="register", is_used=False
                ).delete()

                PendingRegistration.objects.create(
                    username=serializer.validated_data["username"],
                    email=email,
                    password=hashed_password,
                    expires_at=timezone.now() + timedelta(minutes=10),
                )

                EmailOTP.objects.create(
                    email=email,
                    otp=otp,
                    purpose="register",
                    expires_at=timezone.now() + timedelta(minutes=10),
                )

                send_otp_email(email, otp)

            return Response(
                {
                    "message": "OTP sent successfully. Please check your email inbox and spam folder too."
                },
                status=status.HTTP_200_OK,
            )

        except Exception:
            # Don't leak internal error details (DB errors, email
            # provider errors, etc.) to the client.
            logger.exception("Registration OTP flow failed for %s", email)
            return Response(
                {"error": "Could not process registration. Please try again."},
                status=status.HTTP_400_BAD_REQUEST,
            )


class VerifyOTP(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = VerifyOTPSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(
                {"error": serializer.errors}, status=status.HTTP_400_BAD_REQUEST
            )

        email = serializer.validated_data["email"]
        otp = serializer.validated_data["otp"]

        otp_record = EmailOTP.objects.filter(
            email=email, purpose="register", is_used=False
        ).first()

        if otp_record is None:
            return Response(
                {"error": "No active OTP found."}, status=status.HTTP_400_BAD_REQUEST
            )

        # 1. Check expiry first
        if otp_record.expires_at < timezone.now():
            PendingRegistration.objects.filter(email=email).delete()
            otp_record.is_used = True
            otp_record.save(update_fields=["is_used"])
            return Response(
                {"error": "OTP expired. Please register again."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # 2. Check lockout
        if otp_record.verification_attempts >= 5:
            return Response(
                {"error": "Maximum verification attempts reached."},
                status=status.HTTP_403_FORBIDDEN,
            )

        # 3. Validate OTP — attempt counter uses an atomic F() update so
        # two parallel requests can't both read the same count before
        # either writes back (that would let someone dodge the lockout).
        if otp_record.otp != otp:
            EmailOTP.objects.filter(pk=otp_record.pk).update(
                verification_attempts=F("verification_attempts") + 1
            )
            otp_record.refresh_from_db()
            remaining = max(5 - otp_record.verification_attempts, 0)
            return Response(
                {"error": "Invalid OTP.", "remaining_attempts": remaining},
                status=status.HTTP_400_BAD_REQUEST,
            )

        pending_user = PendingRegistration.objects.filter(email=email).first()
        if pending_user is None:
            return Response(
                {"error": "Registration request not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        try:
            with transaction.atomic():
                User.objects.create(
                    username=pending_user.username,
                    email=pending_user.email,
                    password=pending_user.password,
                )
                otp_record.is_used = True
                otp_record.save(update_fields=["is_used"])
                pending_user.delete()

            return Response(
                {"message": "Account created successfully."},
                status=status.HTTP_201_CREATED,
            )

        except Exception:
            logger.exception("Account creation failed for %s", email)
            return Response(
                {"error": "Could not create account. Please try again."},
                status=status.HTTP_400_BAD_REQUEST,
            )


class ForgotPasswordRequestView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = EmailSerializer(data=request.data)

        if not serializer.is_valid():
            return Response(
                {"error": serializer.errors}, status=status.HTTP_400_BAD_REQUEST
            )

        email = serializer.validated_data["email"]

        # Always send back the exact same response whether or not this
        # email belongs to an account. Otherwise this endpoint becomes
        # a free tool for checking who's registered.
        generic_response = Response(
            {
                "message": (
                    "If an account exists for this email, a password "
                    "reset OTP has been sent."
                )
            },
            status=status.HTTP_200_OK,
        )

        if not User.objects.filter(email=email).exists():
            return generic_response

        if not can_send_otp(email) or not check_hourly_limit(email):
            # Still generic — a different response here would tell an
            # attacker the account exists.
            return generic_response

        otp = generate_otp()
        try:
            with transaction.atomic():
                EmailOTP.objects.filter(
                    is_used=False, purpose="reset_password", email=email
                ).delete()
                EmailOTP.objects.create(
                    email=email,
                    otp=otp,
                    purpose="reset_password",
                    expires_at=timezone.now() + timedelta(minutes=10),
                )
            send_otp_email(email, otp)
        except Exception:
            logger.exception("Failed to send password reset OTP for %s", email)

        return generic_response


class ResetPasswordView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):

        serializer = ResetPasswordSerializer(data=request.data)

        if not serializer.is_valid():
            return Response(
                {"error": serializer.errors}, status=status.HTTP_400_BAD_REQUEST
            )

        email = serializer.validated_data["email"]
        otp = serializer.validated_data["otp"]
        new_password = serializer.validated_data["new_password"]

        otp_record = EmailOTP.objects.filter(
            email=email, purpose="reset_password", is_used=False
        ).first()

        if otp_record is None:
            return Response(
                {"error": "Invalid or expired OTP."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Check expiry before attempt count, same order as VerifyOTP.
        if otp_record.expires_at < timezone.now():
            otp_record.is_used = True
            otp_record.save(update_fields=["is_used"])
            return Response(
                {"error": "Invalid or expired OTP."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if otp_record.verification_attempts >= 5:
            return Response(
                {"error": "Maximum verification attempts exceeded."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if otp_record.otp != otp:
            # Same atomic F() update as VerifyOTP — no race condition
            # on the attempt counter.
            EmailOTP.objects.filter(pk=otp_record.pk).update(
                verification_attempts=F("verification_attempts") + 1
            )
            otp_record.refresh_from_db()
            remaining_attempts = max(5 - otp_record.verification_attempts, 0)
            return Response(
                {
                    "error": "Invalid OTP.",
                    "remaining_attempts": remaining_attempts,
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            with transaction.atomic():
                user = User.objects.select_for_update().get(email=email)

                user.set_password(new_password)
                user.save(update_fields=["password"])

                otp_record.is_used = True
                otp_record.save(update_fields=["is_used"])

                # A password reset must kill every existing session.
                # Otherwise, if someone's refresh token was stolen, it
                # would keep working even after they "secure" their
                # account by changing the password.
                RefreshToken.objects.filter(
                    user=user, is_revoked=False
                ).update(is_revoked=True)

            return Response(
                {"message": "Password reset successfully."},
                status=status.HTTP_200_OK,
            )

        except User.DoesNotExist:
            # Kept generic on purpose — same message as "bad OTP".
            return Response(
                {"error": "Invalid or expired OTP."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        except Exception:
            logger.exception("Password reset failed for %s", email)
            return Response(
                {"error": "Could not reset password. Please try again."},
                status=status.HTTP_400_BAD_REQUEST,
            )


class LoginView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):

        serializer = LoginSerializer(data=request.data)

        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        user = serializer.validated_data["user"]

        if user.is_blocked:
            return Response(
                {"error": "This account has been blocked."},
                status=status.HTTP_403_FORBIDDEN,
            )

        access_token = generate_access_token(user)
        refresh_token = generate_refresh_token(user)

        RefreshToken.objects.create(
            user=user,
            token=refresh_token,
            expires_at=timezone.now()
            + timedelta(days=settings.REFRESH_TOKEN_LIFETIME_DAYS),
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

        serializer = RefreshSerializer(data=request.data)

        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        raw_refresh_token = serializer.validated_data["refresh_token"]

        try:
            payload = decode_token(raw_refresh_token)
        except Exception:
            return Response(
                {"error": "Invalid or expired refresh token."},
                status=status.HTTP_401_UNAUTHORIZED,
            )

        if payload.get("type") != "refresh":
            return Response(
                {"error": "Invalid token type."}, status=status.HTTP_400_BAD_REQUEST
            )

        with transaction.atomic():
            try:
                # select_for_update locks this row until the transaction
                # ends, so two simultaneous refresh calls with the same
                # token can't both "win" and both rotate it.
                token_obj = RefreshToken.objects.select_for_update().get(
                    token=raw_refresh_token
                )
            except RefreshToken.DoesNotExist:
                return Response(
                    {"error": "Invalid refresh token."},
                    status=status.HTTP_401_UNAUTHORIZED,
                )

            if token_obj.is_revoked:
                # This token was already rotated (or logged out) once
                # before. Seeing it again means it was copied somewhere
                # it shouldn't have been — kill every active session for
                # this user, not just this one token.
                RefreshToken.objects.filter(
                    user_id=token_obj.user_id, is_revoked=False
                ).update(is_revoked=True)
                return Response(
                    {"error": "Invalid refresh token."},
                    status=status.HTTP_401_UNAUTHORIZED,
                )

            if token_obj.expires_at < timezone.now():
                token_obj.is_revoked = True
                token_obj.save(update_fields=["is_revoked"])
                return Response(
                    {"error": "Refresh token expired. Please login again."},
                    status=status.HTTP_401_UNAUTHORIZED,
                )

            user = token_obj.user

            if user.is_blocked:
                token_obj.is_revoked = True
                token_obj.save(update_fields=["is_revoked"])
                return Response(
                    {"error": "This account has been blocked."},
                    status=status.HTTP_403_FORBIDDEN,
                )

            # Rotate: this token is spent, a new one takes its place.
            token_obj.is_revoked = True
            token_obj.save(update_fields=["is_revoked"])

            new_refresh_token = generate_refresh_token(user)
            RefreshToken.objects.create(
                user=user,
                token=new_refresh_token,
                expires_at=timezone.now()
                + timedelta(days=settings.REFRESH_TOKEN_LIFETIME_DAYS),
            )

        access_token = generate_access_token(user)

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
        serializer = RefreshSerializer(data=request.data)

        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        raw_refresh_token = serializer.validated_data["refresh_token"]

        try:
            token_obj = RefreshToken.objects.get(token=raw_refresh_token)
        except RefreshToken.DoesNotExist:
            return Response(
                {"error": "Refresh token not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        # A user can only log out their own session. We return the
        # same 404 whether the token doesn't exist or belongs to
        # someone else, so this can't be used to probe other users'
        # tokens.
        if token_obj.user_id != request.user.id:
            return Response(
                {"error": "Refresh token not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        token_obj.is_revoked = True
        token_obj.save(update_fields=["is_revoked"])

        return Response({"message": "Logged out successfully."})