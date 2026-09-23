from django.contrib.auth.password_validation import validate_password
from rest_framework import serializers

from .models import PendingRegistration, User


GENERIC_REGISTRATION_ERROR = "Unable to register with these details."
INVALID_CREDENTIALS_ERROR = "Invalid credentials."


class RegisterRequestSerializer(serializers.Serializer):
    username = serializers.CharField(max_length=100)
    email = serializers.EmailField()
    password = serializers.CharField(write_only=True)

    def validate_email(self, value):
        if User.objects.filter(email=value).exists():
            raise serializers.ValidationError(
                GENERIC_REGISTRATION_ERROR
            )

        if PendingRegistration.objects.filter(email=value).exists():
            raise serializers.ValidationError(
                GENERIC_REGISTRATION_ERROR
            )

        return value

    def validate_username(self, value):
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


class LoginSerializer(serializers.Serializer):
    email = serializers.EmailField()
    password = serializers.CharField(write_only=True)

    def validate(self, attrs):
        email = attrs.get("email")
        password = attrs.get("password")

        try:
            user = User.objects.get(email=email)
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

    def validate_new_password(self, value):
        validate_password(value)
        return value