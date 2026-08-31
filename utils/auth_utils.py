import secrets
from datetime import datetime, timedelta, timezone as dt_timezone

import jwt
from django.conf import settings
from rest_framework import status
from rest_framework.response import Response


def _access_token_lifetime():
    return timedelta(minutes=getattr(settings, "ACCESS_TOKEN_LIFETIME_MINUTES", 15))


def _refresh_token_lifetime():
    return timedelta(days=getattr(settings, "REFRESH_TOKEN_LIFETIME_DAYS", 7))


def generate_access_token(user):

    payload = {
        "user_id": user.id,
        "email": user.email,
        "type": "access",
        "iat": datetime.now(dt_timezone.utc),
        "exp": datetime.now(dt_timezone.utc) + _access_token_lifetime(),
    }

    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm="HS256")


def generate_refresh_token(user):

    payload = {
        "user_id": user.id,
        "type": "refresh",
        # A random id per token. Not required for the hash-lookup flow
        # to work, but it means two refresh tokens minted in the same
        # second for the same user still hash to different values.
        "jti": secrets.token_hex(16),
        "iat": datetime.now(dt_timezone.utc),
        "exp": datetime.now(dt_timezone.utc) + _refresh_token_lifetime(),
    }

    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm="HS256")


def decode_token(token):

    return jwt.decode(
        token,
        settings.JWT_SECRET_KEY,
        algorithms=["HS256"],
    )


def admin_required(user):

    if user.role != "admin":

        return Response(
            {
                "error": "Admin access required"
            },
            status=status.HTTP_403_FORBIDDEN
        )

    return None