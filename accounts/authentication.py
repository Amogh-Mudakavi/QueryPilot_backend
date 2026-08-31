from django.contrib.auth import get_user_model
from rest_framework.authentication import BaseAuthentication
from rest_framework.exceptions import AuthenticationFailed

from utils.auth_utils import decode_token


class JWTAuthentication(BaseAuthentication):
    keyword = "Bearer"

    def authenticate(self, request):
        auth_header = request.headers.get("Authorization")

        if not auth_header:
            return None

        parts = auth_header.split(" ", 1)
        if len(parts) != 2 or parts[0].lower() != self.keyword.lower():
            raise AuthenticationFailed("Invalid authorization header format.")

        token = parts[1].strip()
        if not token:
            raise AuthenticationFailed("Invalid authorization header format.")

        try:
            payload = decode_token(token)
        except Exception as exc:
            raise AuthenticationFailed("Invalid or expired token.") from exc

        # A refresh token is also a validly-signed JWT. Without this
        # check, someone could take a refresh token and use it directly
        # as a bearer token on any protected endpoint — completely
        # skipping the short access-token lifetime.
        if payload.get("type") != "access":
            raise AuthenticationFailed("Invalid token type.")

        user_id = payload.get("user_id")
        if not user_id:
            raise AuthenticationFailed("Invalid token payload.")

        User = get_user_model()
        try:
            user = User.objects.get(pk=user_id)
        except User.DoesNotExist as exc:
            raise AuthenticationFailed("User not found.") from exc

        if getattr(user, "is_blocked", False):
            raise AuthenticationFailed("User is blocked.")

        return user, token