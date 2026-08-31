from django.db import models
from django.contrib.auth.models import AbstractUser


class User(AbstractUser):
    name = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    email = models.EmailField(unique=True)

    ROLE_CHOICES = [
        ("customer", "customer"),
        ("admin", "admin"),
    ]

    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default="customer")
    is_blocked = models.BooleanField(default=False)

    def __str__(self):
        return self.email


class RefreshToken(models.Model):

    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="refresh_tokens"
    )

    token = models.CharField(max_length=512, unique=True)

    created_at = models.DateTimeField(auto_now_add=True)

    expires_at = models.DateTimeField()

    is_revoked = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.user.email}"


class EmailOTP(models.Model):

    PURPOSE_CHOICES = [
        ("register", "Register"),
        ("reset_password", "Reset Password"),
    ]

    email = models.EmailField()

    otp = models.CharField(max_length=6)

    purpose = models.CharField(max_length=20, choices=PURPOSE_CHOICES)

    expires_at = models.DateTimeField()

    is_used = models.BooleanField(default=False)

    verification_attempts = models.IntegerField(default=0)

    last_sent_ip = models.GenericIPAddressField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'{self.email} - {self.purpose}'


class PendingRegistration(models.Model):

    username = models.CharField(max_length=100)

    email = models.EmailField(unique=True)

    password = models.CharField(max_length=255)

    is_verified = models.BooleanField(default=False)

    expires_at = models.DateTimeField()

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.email