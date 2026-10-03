"""Tests module for QueryPilot."""

from datetime import timedelta
from unittest.mock import patch

import jwt
from django.contrib.auth.hashers import check_password
from django.test import TestCase
from django.utils import timezone
from rest_framework.test import APIClient

from utils.auth_utils import (
    decode_token,
    generate_access_token,
    generate_refresh_token,
    hash_refresh_token,
)
from utils.email_utils import (
    generate_otp,
    hash_otp,
    verify_otp,
)

from .models import EmailOTP, RefreshToken, User


class AuthSecurityTests(TestCase):
    def setUp(self):
        self.client = APIClient()

        self.user = User.objects.create(
            username="testuser",
            email="test@example.com",
            name="Test User",
        )

        self.user.set_password("StrongPassword123!")

        self.user.save()

    def test_otp_is_hashed(self):
        otp = generate_otp()
        otp_hash = hash_otp(otp)

        self.assertNotEqual(
            otp,
            otp_hash,
        )

        self.assertTrue(
            verify_otp(
                otp,
                otp_hash,
            )
        )

        self.assertFalse(
            verify_otp(
                "000000",
                otp_hash,
            )
        )

    def test_refresh_token_is_stored_as_hash(self):
        raw_token = generate_refresh_token(self.user)

        stored_hash = hash_refresh_token(raw_token)

        refresh_record = RefreshToken.objects.create(
            user=self.user,
            token=stored_hash,
            expires_at=timezone.now() + timedelta(days=7),
        )

        self.assertNotEqual(
            refresh_record.token,
            raw_token,
        )

        self.assertEqual(
            len(refresh_record.token),
            64,
        )

        self.assertEqual(
            refresh_record.token,
            hash_refresh_token(raw_token),
        )

    def test_access_token_contains_access_type(self):
        token = generate_access_token(self.user)

        payload = decode_token(token)

        self.assertEqual(
            payload["type"],
            "access",
        )

        self.assertEqual(
            payload["user_id"],
            self.user.id,
        )

    def test_refresh_token_contains_refresh_type(self):
        token = generate_refresh_token(self.user)

        payload = decode_token(token)

        self.assertEqual(
            payload["type"],
            "refresh",
        )

        self.assertEqual(
            payload["user_id"],
            self.user.id,
        )

        self.assertIn(
            "jti",
            payload,
        )

    def test_inactive_user_cannot_authenticate(self):
        self.user.is_active = False
        self.user.save(update_fields=["is_active"])

        token = generate_access_token(self.user)

        response = self.client.get(
            "/api/auth/me/",
            HTTP_AUTHORIZATION=f"Bearer {token}",
        )

        self.assertEqual(
            response.status_code,
            401,
        )

    def test_blocked_user_cannot_authenticate(self):
        self.user.is_blocked = True
        self.user.save(update_fields=["is_blocked"])

        token = generate_access_token(self.user)

        response = self.client.get(
            "/api/auth/me/",
            HTTP_AUTHORIZATION=f"Bearer {token}",
        )

        self.assertEqual(
            response.status_code,
            401,
        )

    @patch(
        "accounts.views.login_rate_limited",
        return_value=False,
    )
    @patch(
        "accounts.views.clear_login_rate_limits",
    )
    def test_login_returns_tokens(
        self,
        mock_clear_limits,
        mock_rate_limited,
    ):
        response = self.client.post(
            "/api/auth/login/",
            {
                "email": "test@example.com",
                "password": "StrongPassword123!",
            },
            format="json",
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        self.assertIn(
            "access_token",
            response.data,
        )

        self.assertIn(
            "refresh_token",
            response.data,
        )

        self.assertTrue(RefreshToken.objects.filter(user=self.user).exists())

        stored_token = RefreshToken.objects.filter(user=self.user).latest("created_at")

        self.assertNotEqual(
            stored_token.token,
            response.data["refresh_token"],
        )

        self.assertEqual(
            stored_token.token,
            hash_refresh_token(response.data["refresh_token"]),
        )

    @patch(
        "accounts.views.login_rate_limited",
        return_value=False,
    )
    def test_invalid_login_does_not_return_tokens(
        self,
        mock_rate_limited,
    ):
        response = self.client.post(
            "/api/auth/login/",
            {
                "email": "test@example.com",
                "password": "WrongPassword123!",
            },
            format="json",
        )

        self.assertEqual(
            response.status_code,
            400,
        )

        self.assertNotIn(
            "access_token",
            response.data,
        )

        self.assertNotIn(
            "refresh_token",
            response.data,
        )

    @patch(
        "accounts.views.login_rate_limited",
        return_value=True,
    )
    def test_login_rate_limit_returns_429(
        self,
        mock_rate_limited,
    ):
        response = self.client.post(
            "/api/auth/login/",
            {
                "email": "test@example.com",
                "password": "StrongPassword123!",
            },
            format="json",
        )

        self.assertEqual(
            response.status_code,
            429,
        )

    def test_expired_access_token_is_rejected(self):
        now = timezone.now()

        payload = {
            "user_id": self.user.id,
            "email": self.user.email,
            "type": "access",
            "iat": now - timedelta(minutes=30),
            "exp": now - timedelta(minutes=1),
        }

        from django.conf import settings

        token = jwt.encode(
            payload,
            settings.JWT_SECRET_KEY,
            algorithm=settings.JWT_ALGORITHM,
        )

        response = self.client.get(
            "/api/auth/me/",
            HTTP_AUTHORIZATION=f"Bearer {token}",
        )

        self.assertEqual(
            response.status_code,
            401,
        )

    def test_refresh_token_has_unique_jti(self):
        token_one = generate_refresh_token(self.user)

        token_two = generate_refresh_token(self.user)

        payload_one = decode_token(token_one)

        payload_two = decode_token(token_two)

        self.assertNotEqual(
            payload_one["jti"],
            payload_two["jti"],
        )

    def test_single_use_otp(self):
        otp = generate_otp()

        record = EmailOTP.objects.create(
            email=self.user.email,
            otp=hash_otp(otp),
            purpose="reset_password",
            expires_at=timezone.now() + timedelta(minutes=10),
        )

        self.assertFalse(record.is_used)

        record.is_used = True

        record.save(update_fields=["is_used"])

        record.refresh_from_db()

        self.assertTrue(record.is_used)
