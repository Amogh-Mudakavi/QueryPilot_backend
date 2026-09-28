from django.contrib.auth.password_validation import validate_password
from django.utils import timezone
from rest_framework import serializers

from .models import EmailOTP, PendingRegistration, User


GENERIC_REGISTRATION_ERROR = "Unable to register with these details."
INVALID_CREDENTIALS_ERROR = "Invalid credentials."


class RegisterRequestSerializer(serializers.Serializer):
    username = serializers.CharField(max_length=100)
    email = serializers.EmailField()
    password = serializers.CharField(write_only=True)

    def validate_email(self, value):
        # Keep email identity consistent across register/login/resend flows.
        value = value.strip().lower()

        if User.objects.filter(email__iexact=value).exists():
            raise serializers.ValidationError(
                GENERIC_REGISTRATION_ERROR
            )

        # A registration draft is temporary state. Once it expires, it must
        # not permanently block the user from registering again.
        pending_qs = PendingRegistration.objects.filter(
            email__iexact=value
        )
        now = timezone.now()

        pending_qs.filter(
            expires_at__lte=now
        ).delete()

        EmailOTP.objects.filter(
            email__iexact=value,
            purpose="register",
            is_used=False,
            expires_at__lte=now,
        ).update(is_used=True)

        if pending_qs.filter(expires_at__gt=now).exists():
            raise serializers.ValidationError(
                GENERIC_REGISTRATION_ERROR
            )

        return value

    def validate_username(self, value):
        value = value.strip()

        if User.objects.filter(username=value).exists():
            raise serializers.ValidationError(
                GENERIC_REGISTRATION_ERROR
            )

        return value

    def validate_password(self, value):
        validate_password(value)
        return value


class VerifyOTPSerializer(serializers.Serializer):
    email = serializers.EmailField()
    otp = serializers.CharField(
        min_length=6,
        max_length=6,
        trim_whitespace=True,
    )

    def validate_email(self, value):
        return value.strip().lower()


class LoginSerializer(serializers.Serializer):
    email = serializers.EmailField()
    password = serializers.CharField(write_only=True)

    def validate(self, attrs):
        email = attrs.get("email")
        password = attrs.get("password")

        if isinstance(email, str):
            email = email.strip().lower()
            attrs["email"] = email

        try:
            user = User.objects.filter(
                email__iexact=email
            ).first()

            if user is None:
                raise User.DoesNotExist
        except User.DoesNotExist:
            raise serializers.ValidationError(
                INVALID_CREDENTIALS_ERROR
            )

        if not user.is_active or user.is_blocked:
            raise serializers.ValidationError(
                INVALID_CREDENTIALS_ERROR
            )

        if not user.check_password(password):
            raise serializers.ValidationError(
                INVALID_CREDENTIALS_ERROR
            )

        attrs["user"] = user
        return attrs


class RefreshSerializer(serializers.Serializer):
    refresh_token = serializers.CharField(
        allow_blank=False,
        trim_whitespace=True,
    )


class EmailSerializer(serializers.Serializer):
    email = serializers.EmailField()

    def validate_email(self, value):
        return value.strip().lower()


class ResetPasswordSerializer(serializers.Serializer):
    email = serializers.EmailField()
    otp = serializers.CharField(
        min_length=6,
        max_length=6,
        trim_whitespace=True,
    )
    new_password = serializers.CharField(
        write_only=True
    )

    def validate_email(self, value):
        return value.strip().lower()

    def validate_new_password(self, value):
        validate_password(value)
        return value