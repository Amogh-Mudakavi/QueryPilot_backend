from datetime import timedelta
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
    RefreshSerialier,
    RegisterRequestSerializer,
    ResetPasswordSerializer,
    VerifyOTPSerializer,
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

                # FIX 1: purpose was "Register" (capital R), model defines it as "register"
                EmailOTP.objects.filter(
                    email=email, purpose="register", is_used=False
                ).delete()

                PendingRegistration.objects.create(
                    username=serializer.validated_data["username"],
                    email=email,
                    password=hashed_password,
                    expires_at=timezone.now() + timedelta(minutes=10),
                )

                # FIX 1: purpose was "Register" (capital R), model defines it as "register"
                EmailOTP.objects.create(
                    email=email,
                    otp=otp,
                    purpose="register",
                    expires_at=timezone.now() + timedelta(minutes=10),
                )

                send_otp_email(email, otp)

            return Response(
                {
                    "message": "OTP sent successfully. Please Check your email inbox and spam folder too."
                },
                status=status.HTTP_200_OK,
            )

        except Exception as e:

            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)


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

        # FIX 1: purpose was "Register" (capital R), model defines it as "register"
        otp_record = EmailOTP.objects.filter(
            email=email, purpose="register", is_used=False
        ).first()

        if otp_record is None:
            return Response(
                {"error": "No active OTP found"}, status=status.HTTP_400_BAD_REQUEST
            )

        # 1. Check expiry first
        if otp_record.expires_at < timezone.now():
            PendingRegistration.objects.filter(email=email).delete()
            otp_record.is_used = True
            otp_record.save()
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

        # 3. Validate OTP
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
                new_user = User.objects.create(
                    username=pending_user.username,
                    email=pending_user.email,
                    password=pending_user.password,
                )
                otp_record.is_used = True
                otp_record.save()
                pending_user.delete()

            return Response(
                {"message": "Account created successfully."},
                status=status.HTTP_201_CREATED,
            )

        except Exception as e:
            return Response({"errors": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class ForgotPasswordRequestView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = EmailSerializer(data=request.data)

        if not serializer.is_valid():
            return Response(
                {"errors": serializer.errors}, status=status.HTTP_400_BAD_REQUEST
            )

        email = serializer.validated_data["email"]

        if not User.objects.filter(email=email).exists():
            return Response(
                {"errors": "Theres no email by this id , Please Register"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if not can_send_otp(email):
            return Response(
                {"errors": "Wait until Cooldown"}, status=status.HTTP_400_BAD_REQUEST
            )

        if not check_hourly_limit(email):
            return Response(
                {"error": "Maximum OTP limit reached. Please try again later."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        otp = generate_otp()
        try:
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

            return Response(
                {"message": "Passwrod reset OTP sent successfully"},
                status=status.HTTP_200_OK,
            )

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)


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
                {"error": "No active OTP found"}, status=status.HTTP_400_BAD_REQUEST
            )

        if otp_record.verification_attempts >= 5:
            return Response(
                {"error": "Maximum verification attempts exceeded"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if otp_record.expires_at < timezone.now():
            otp_record.is_used = True
            otp_record.save()
            return Response(
                {"error": "OTP expired"}, status=status.HTTP_400_BAD_REQUEST
            )

        if otp_record.otp != otp:

            otp_record.verification_attempts += 1
            otp_record.save()

            remaining_attempts = max(5 - otp_record.verification_attempts, 0)

            return Response(
                {"error": "Invalid OTP", "remaining_attempts": remaining_attempts},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:

            user = User.objects.get(email=email)

            user.set_password(new_password)
            user.save()

            otp_record.is_used = True
            otp_record.save()

            return Response(
                {"message": "Password reset successfully"}, status=status.HTTP_200_OK
            )

        except User.DoesNotExist:

            return Response(
                {"error": "User not found"}, status=status.HTTP_404_NOT_FOUND
            )

        except Exception as e:

            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class LoginView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):

        serializer = LoginSerializer(data=request.data)

        if serializer.is_valid():

            user = serializer.validated_data["user"]

            access_token = generate_access_token(user)

            refresh_token = generate_refresh_token(user)
            RefreshToken.objects.create(
                user=user,
                token=refresh_token,
                expires_at=timezone.now() + timedelta(days=7),
            )


            return Response(
                {
                    "access_token": access_token,
                    "refresh_token": refresh_token,
                },
                status=status.HTTP_200_OK,
            )

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class getme(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):

        try:

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

        except Exception as e:

            return Response(
                {"error": str(e)},
                status=status.HTTP_401_UNAUTHORIZED,
            )


class RefreshView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):

        serializer = RefreshSerialier(data=request.data)

        if not serializer.is_valid():

            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        refresh_token = serializer.validated_data["refresh_token"]
        try:

            payload = decode_token(refresh_token)

        except Exception as e:

            return Response(
                {"error": str(e)}, status=status.HTTP_401_UNAUTHORIZED
            )

        # FIX 2: was raising ValidationError which causes 500, return Response instead
        if payload["type"] != "refresh":
            return Response(
                {"error": "Invalid token type"}, status=status.HTTP_400_BAD_REQUEST
            )

        try:

            refresh_token_obj = RefreshToken.objects.get(token=refresh_token)
            if refresh_token_obj.expires_at < timezone.now():

                refresh_token_obj.is_revoked = True
                refresh_token_obj.save()

                # FIX 2: was raising ValidationError which causes 500, return Response instead
                return Response(
                    {"error": "Refresh token expired. Please login again."},
                    status=status.HTTP_401_UNAUTHORIZED,
                )

        except RefreshToken.DoesNotExist:
            # FIX 2: was raising ValidationError which causes 500, return Response instead
            return Response(
                {"error": "Refresh token not found"}, status=status.HTTP_404_NOT_FOUND
            )

        if refresh_token_obj.is_revoked:
            # FIX 2: was raising ValidationError which causes 500, return Response instead
            return Response(
                {"error": "Token revoked. Please login again."},
                status=status.HTTP_401_UNAUTHORIZED,
            )

        user = refresh_token_obj.user

        access_token = generate_access_token(user)

        return Response({"access_token": access_token}, status=status.HTTP_200_OK)


class LogoutView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serialier = RefreshSerialier(data=request.data)

        if not serialier.is_valid():
            return Response(serialier.errors, status=status.HTTP_400_BAD_REQUEST)

        refresh_token = serialier.validated_data["refresh_token"]

        try:

            refresh_token_obj = RefreshToken.objects.get(token=refresh_token)

        except RefreshToken.DoesNotExist:

            return Response(
                {"error": "Refresh token not found"}, status=status.HTTP_404_NOT_FOUND
            )

        refresh_token_obj.is_revoked = True
        refresh_token_obj.save()


        return Response({"message": "logged out successfully"})