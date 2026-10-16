"""Models module for QueryPilot."""

from django.contrib.auth.models import AbstractUser
from django.db import models


class User(AbstractUser):
    name = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    email = models.EmailField(unique=True)

    ROLE_CHOICES = [
        ("customer", "customer"),
        ("admin", "admin"),
    ]

    role = models.CharField(
        max_length=20,
        choices=ROLE_CHOICES,
        default="customer",
    )

    is_blocked = models.BooleanField(default=False)

    def __str__(self):
        return self.email


class RefreshToken(models.Model):
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="refresh_tokens",
    )

    token = models.CharField(
        max_length=64,
        unique=True,
    )

    created_at = models.DateTimeField(auto_now_add=True)

    expires_at = models.DateTimeField()

    is_revoked = models.BooleanField(default=False)

    def __str__(self):
        return "RefreshToken"


class EmailOTP(models.Model):
    PURPOSE_CHOICES = [
        ("register", "Register"),
        ("reset_password", "Reset Password"),
    ]

    email = models.EmailField()

    otp = models.CharField(max_length=128)

    purpose = models.CharField(
        max_length=20,
        choices=PURPOSE_CHOICES,
    )

    expires_at = models.DateTimeField()

    is_used = models.BooleanField(default=False)

    verification_attempts = models.PositiveSmallIntegerField(default=0)

    last_sent_ip = models.GenericIPAddressField(
        null=True,
        blank=True,
    )

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"EmailOTP ({self.purpose})"


class PendingRegistration(models.Model):
    username = models.CharField(max_length=100)

    email = models.EmailField(unique=True)

    password = models.CharField(max_length=255)

    is_verified = models.BooleanField(default=False)

    expires_at = models.DateTimeField()

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return "PendingRegistration"

class GoogleIdentity(models.Model):
    """
    Maps a Google account to a QueryPilot user.
    Google subject ID is the stable identity key, not the email.
    """

    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE,
        related_name="google_identity",
    )

    google_subject = models.CharField(
        max_length=255,
        unique=True,
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"GoogleIdentity (user={self.user_id})"


class OAuthLoginCode(models.Model):
    """
    Stores a hashed, short-lived, single-use code for JWT handoff.
    The raw code must never be stored in the database.
    """

    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="oauth_login_codes",
    )

    code_hash = models.CharField(
        max_length=64,
        unique=True,
    )

    created_at = models.DateTimeField(auto_now_add=True)

    expires_at = models.DateTimeField()

    consumed_at = models.DateTimeField(
        null=True,
        blank=True,
    )

    def __str__(self):
        return f"OAuthLoginCode (user={self.user_id})"