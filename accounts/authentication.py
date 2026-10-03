"""Authentication module for QueryPilot."""

import jwt
from django.contrib.auth import get_user_model
from rest_framework import authentication
from rest_framework.exceptions import AuthenticationFailed

from utils.auth_utils import decode_token

User = get_user_model()


class JWTAuthentication(authentication.BaseAuthentication):
    def authenticate(self, request):
        auth_header = request.headers.get("Authorization")

        if not auth_header:
            return None

        parts = auth_header.split()

        if len(parts) != 2 or parts[0].lower() != "bearer":
            raise AuthenticationFailed("Invalid authentication credentials.")

        token = parts[1].strip()

        if not token:
            raise AuthenticationFailed("Invalid authentication credentials.")

        try:
            payload = decode_token(token)

        except jwt.ExpiredSignatureError:
            raise AuthenticationFailed("Authentication token has expired.")

        except jwt.InvalidTokenError:
            raise AuthenticationFailed("Invalid authentication credentials.")

        if payload.get("type") != "access":
            raise AuthenticationFailed("Invalid authentication credentials.")

        user_id = payload.get("user_id")

        if not user_id:
            raise AuthenticationFailed("Invalid authentication credentials.")

        try:
            user = User.objects.get(id=user_id)
        except User.DoesNotExist:
            raise AuthenticationFailed("Invalid authentication credentials.")

        if not user.is_active or user.is_blocked:
            raise AuthenticationFailed("Invalid authentication credentials.")

        return (user, token)

    def authenticate_header(self, request):
        return "Bearer"
