import hashlib
import secrets
from datetime import datetime, timedelta, timezone as dt_timezone

import jwt
from django.conf import settings
from django.core.cache import cache
from rest_framework import status
from rest_framework.response import Response


def _access_token_lifetime():
    return timedelta(
        minutes=settings.ACCESS_TOKEN_LIFETIME_MINUTES
    )


def _refresh_token_lifetime():
    return timedelta(
        days=settings.REFRESH_TOKEN_LIFETIME_DAYS
    )


def generate_access_token(user):
    now = datetime.now(dt_timezone.utc)

    payload = {
        "user_id": user.id,
        "email": user.email,
        "type": "access",
        "iat": now,
        "exp": now + _access_token_lifetime(),
    }

    return jwt.encode(
        payload,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )


def refresh_token_expires_at():
    return (
        datetime.now(dt_timezone.utc)
        + _refresh_token_lifetime()
    )


def generate_refresh_token(user):
    now = datetime.now(dt_timezone.utc)

    payload = {
        "user_id": user.id,
        "type": "refresh",
        "jti": secrets.token_hex(16),
        "iat": now,
        "exp": now + _refresh_token_lifetime(),
    }

    return jwt.encode(
        payload,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )


def decode_token(token):
    return jwt.decode(
        token,
        settings.JWT_SECRET_KEY,
        algorithms=[settings.JWT_ALGORITHM],
    )


def hash_refresh_token(token):
    return hashlib.sha256(
        token.encode("utf-8")
    ).hexdigest()


def _login_key(prefix, value):
    digest = hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()

    return f"querypilot:auth:login:{prefix}:{digest}"


def _consume_rate_limit(
    key,
    limit,
    window_seconds,
):
    try:
        created = cache.add(
            key,
            1,
            timeout=window_seconds,
        )

        if created:
            return True

        return cache.incr(key) <= limit

    except Exception:
        # Authentication must remain available if the cache is
        # temporarily unavailable. The Redis-backed rate limit is
        # an abuse-control layer, not the credential check itself.
        return True


def login_rate_limited(
    email,
    ip_address=None,
):
    email_key = _login_key(
        "email",
        email.lower().strip(),
    )

    if not _consume_rate_limit(
        email_key,
        settings.LOGIN_RATE_LIMIT_EMAIL_ATTEMPTS,
        settings.LOGIN_RATE_LIMIT_WINDOW_SECONDS,
    ):
        return True

    if ip_address:
        ip_key = _login_key(
            "ip",
            ip_address,
        )

        if not _consume_rate_limit(
            ip_key,
            settings.LOGIN_RATE_LIMIT_IP_ATTEMPTS,
            settings.LOGIN_RATE_LIMIT_WINDOW_SECONDS,
        ):
            return True

    return False


def clear_login_rate_limits(
    email,
    ip_address=None,
):
    keys = [
        _login_key(
            "email",
            email.lower().strip(),
        )
    ]

    if ip_address:
        keys.append(
            _login_key(
                "ip",
                ip_address,
            )
        )

    try:
        cache.delete_many(keys)
    except Exception:
        pass


def admin_required(user):
    if user.role != "admin":
        return Response(
            {"error": "Admin access required"},
            status=status.HTTP_403_FORBIDDEN,
        )

    return None