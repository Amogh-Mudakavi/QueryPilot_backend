"""Google OAuth views for QueryPilot."""

import hashlib
import logging
import secrets
import uuid
from datetime import timedelta
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests
from django.conf import settings
from django.contrib.auth import get_user_model
from django.db import IntegrityError, transaction
from django.http import HttpResponseRedirect, HttpResponseServerError
from django.utils import timezone
from django.utils.text import slugify
from google.auth.transport.requests import Request as GoogleRequest
from google.oauth2 import id_token
from rest_framework import status
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView

from utils.auth_utils import (
    generate_access_token,
    generate_refresh_token,
    hash_refresh_token,
    refresh_token_expires_at,
)

from .models import GoogleIdentity, OAuthLoginCode, RefreshToken

logger = logging.getLogger(__name__)

User = get_user_model()

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"


def _oauth_configured():
    """Return whether the required Google OAuth settings are present."""
    return all(
        [
            settings.GOOGLE_CLIENT_ID,
            settings.GOOGLE_CLIENT_SECRET,
            settings.GOOGLE_REDIRECT_URI,
            settings.OAUTH_FRONTEND_CALLBACK_URL,
        ]
    )


def _frontend_redirect(**params):
    """Redirect only to the frontend callback configured by the application."""
    callback_url = settings.OAUTH_FRONTEND_CALLBACK_URL

    if not callback_url:
        logger.error("OAuth frontend callback URL is not configured.")
        return HttpResponseServerError("OAuth is not configured.")

    parts = urlsplit(callback_url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query.update(params)

    destination = urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            parts.path,
            urlencode(query),
            parts.fragment,
        )
    )

    return HttpResponseRedirect(destination)


def _create_oauth_login_code(user):
    """Create a short-lived login code and store only its SHA-256 hash."""
    raw_code = secrets.token_urlsafe(32)
    code_hash = hashlib.sha256(raw_code.encode("utf-8")).hexdigest()

    lifetime = settings.OAUTH_LOGIN_CODE_LIFETIME_SECONDS

    OAuthLoginCode.objects.create(
        user=user,
        code_hash=code_hash,
        expires_at=timezone.now() + timedelta(seconds=lifetime),
    )

    return raw_code


def _new_google_username(email):
    """Generate a unique username for a Google-created account."""
    local_part = email.split("@", 1)[0]
    base = slugify(local_part)[:120] or "google_user"

    return f"{base}_{uuid.uuid4().hex[:12]}"


class GoogleOAuthStartView(APIView):
    """Start Google signup or signin."""

    permission_classes = [AllowAny]
    authentication_classes = []

    def get(self, request):
        if not _oauth_configured():
            logger.error("Google OAuth configuration is incomplete.")
            return Response(
                {"error": "Google sign-in is not configured."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        mode = request.query_params.get("mode", "")

        if mode not in {"signup", "signin"}:
            return Response(
                {"error": "Mode must be signup or signin."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        state = secrets.token_urlsafe(32)
        session_key = f"google_oauth_{state}"

        request.session[session_key] = {
            "state": state,
            "mode": mode,
            "created_at": int(timezone.now().timestamp()),
        }

        params = {
            "client_id": settings.GOOGLE_CLIENT_ID,
            "redirect_uri": settings.GOOGLE_REDIRECT_URI,
            "response_type": "code",
            "scope": "openid email profile",
            "state": state,
            "prompt": "select_account",
        }

        authorization_url = f"{GOOGLE_AUTH_URL}?{urlencode(params)}"

        return HttpResponseRedirect(authorization_url)


class GoogleOAuthCallbackView(APIView):
    """
    Validate Google's response and create/find the QueryPilot account.

    If Google returns a verified email matching an existing QueryPilot
    account created through the verified-email registration flow, link
    the Google identity to that existing account instead of creating a
    duplicate account or rejecting the login.
    """

    permission_classes = [AllowAny]
    authentication_classes = []

    def get(self, request):
        if not _oauth_configured():
            logger.error("Google OAuth configuration is incomplete.")
            return HttpResponseServerError("OAuth is not configured.")

        state = request.query_params.get("state", "")

        if not state or len(state) > 200:
            return _frontend_redirect(error="invalid_state")

        session_key = f"google_oauth_{state}"
        state_data = request.session.get(session_key)

        # Consume the state even if the callback is otherwise invalid.
        if state_data is not None:
            del request.session[session_key]

        if not isinstance(state_data, dict):
            return _frontend_redirect(error="invalid_state")

        expected_state = state_data.get("state", "")

        if not secrets.compare_digest(state, expected_state):
            return _frontend_redirect(error="invalid_state")

        created_at = state_data.get("created_at")
        now = timezone.now()
        state_lifetime = settings.OAUTH_STATE_LIFETIME_SECONDS

        if (
            not isinstance(created_at, int)
            or created_at > int(now.timestamp()) + 30
            or int(now.timestamp()) - created_at > state_lifetime
        ):
            return _frontend_redirect(error="expired_state")

        mode = state_data.get("mode")

        if mode not in {"signup", "signin"}:
            return _frontend_redirect(error="invalid_state")

        if request.query_params.get("error"):
            # Do not forward arbitrary provider error text to the frontend.
            return _frontend_redirect(error="google_login_cancelled")

        authorization_code = request.query_params.get("code", "")

        if not authorization_code:
            return _frontend_redirect(error="google_login_failed")

        try:
            token_response = requests.post(
                GOOGLE_TOKEN_URL,
                data={
                    "code": authorization_code,
                    "client_id": settings.GOOGLE_CLIENT_ID,
                    "client_secret": settings.GOOGLE_CLIENT_SECRET,
                    "redirect_uri": settings.GOOGLE_REDIRECT_URI,
                    "grant_type": "authorization_code",
                },
                timeout=10,
            )
            token_response.raise_for_status()
            token_data = token_response.json()

            raw_id_token = token_data.get("id_token")

            if not raw_id_token:
                return _frontend_redirect(error="google_login_failed")

            claims = id_token.verify_oauth2_token(
                raw_id_token,
                GoogleRequest(),
                settings.GOOGLE_CLIENT_ID,
            )

        except (
            requests.RequestException,
            ValueError,
            KeyError,
        ):
            logger.warning("Google OAuth token exchange failed.")
            return _frontend_redirect(error="google_login_failed")

        except Exception:
            # Includes ID-token validation failures.
            logger.warning("Google OAuth identity validation failed.")
            return _frontend_redirect(error="google_login_failed")

        google_subject = claims.get("sub")
        email = claims.get("email")
        email_verified = claims.get("email_verified") is True

        if (
            not isinstance(google_subject, str)
            or not google_subject
            or not isinstance(email, str)
            or not email
            or not email_verified
        ):
            return _frontend_redirect(error="google_identity_invalid")

        email = email.strip().lower()
        google_name = claims.get("name")

        name = (
            google_name.strip()[:100]
            if isinstance(google_name, str) and google_name.strip()
            else email.split("@", 1)[0][:100]
        )

        try:
            with transaction.atomic():
                # First, find the stable Google identity.
                identity = (
                    GoogleIdentity.objects.select_related("user")
                    .filter(google_subject=google_subject)
                    .first()
                )

                if identity is not None:
                    # Normal Google sign-in for an already-linked identity.
                    user = identity.user

                else:
                    # No Google identity exists for this Google subject.
                    # Look for an existing QueryPilot account by email.
                    existing_user = (
                        User.objects.select_for_update()
                        .filter(email__iexact=email)
                        .first()
                    )

                    if existing_user is not None:
                        # Do not link Google to a disabled account.
                        if not existing_user.is_active or existing_user.is_blocked:
                            return _frontend_redirect(
                                error="account_unavailable"
                            )

                        # A QueryPilot user can have only one Google identity.
                        # Never replace an identity linked to another Google subject.
                        existing_identity = (
                            GoogleIdentity.objects.filter(
                                user=existing_user
                            ).first()
                        )

                        if existing_identity is not None:
                            return _frontend_redirect(
                                error="google_account_already_linked"
                            )

                        # The normal email/password registration flow creates
                        # User only after successful email OTP verification.
                        #
                        # Google has also returned email_verified=True above.
                        # Link this Google identity to the existing user instead
                        # of creating a duplicate QueryPilot account.
                        GoogleIdentity.objects.create(
                            user=existing_user,
                            google_subject=google_subject,
                        )

                        user = existing_user

                    else:
                        # A signin attempt must not create a new account.
                        if mode == "signin":
                            return _frontend_redirect(
                                error="google_account_not_found"
                            )

                        # No QueryPilot account exists. Create a new account
                        # only when the user explicitly chooses Google signup.
                        user = User(
                            username=_new_google_username(email),
                            email=email,
                            name=name,
                        )
                        user.set_unusable_password()
                        user.save()

                        GoogleIdentity.objects.create(
                            user=user,
                            google_subject=google_subject,
                        )

        except IntegrityError:
            # Handle concurrent requests without creating duplicate identities.
            identity = (
                GoogleIdentity.objects.select_related("user")
                .filter(google_subject=google_subject)
                .first()
            )

            if identity is not None:
                user = identity.user

            else:
                # A different Google identity may have been linked to this
                # email while this request was being processed.
                existing_identity = (
                    GoogleIdentity.objects.filter(
                        user__email__iexact=email
                    ).first()
                )

                if existing_identity is not None:
                    return _frontend_redirect(
                        error="google_account_already_linked"
                    )

                logger.warning("Google OAuth account creation conflict.")
                return _frontend_redirect(
                    error="account_creation_failed"
                )

        except Exception:
            logger.exception("Google OAuth account processing failed.")
            return _frontend_redirect(error="google_login_failed")

        # Recheck account status before issuing a login code.
        if not user.is_active or user.is_blocked:
            return _frontend_redirect(error="account_unavailable")

        try:
            raw_code = _create_oauth_login_code(user)
        except Exception:
            logger.exception("Could not create OAuth login code.")
            return _frontend_redirect(error="google_login_failed")

        return _frontend_redirect(code=raw_code, mode=mode)


class OAuthCodeExchangeView(APIView):
    """Exchange a single-use OAuth code for QueryPilot JWTs."""

    permission_classes = [AllowAny]
    authentication_classes = []

    def post(self, request):
        raw_code = request.data.get("code", "")

        if (
            not isinstance(raw_code, str)
            or not raw_code
            or len(raw_code) > 200
        ):
            return Response(
                {"error": "Invalid or expired OAuth code."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        code_hash = hashlib.sha256(raw_code.encode("utf-8")).hexdigest()
        now = timezone.now()

        with transaction.atomic():
            try:
                login_code = (
                    OAuthLoginCode.objects.select_for_update()
                    .select_related("user")
                    .get(code_hash=code_hash)
                )
            except OAuthLoginCode.DoesNotExist:
                return Response(
                    {"error": "Invalid or expired OAuth code."},
                    status=status.HTTP_401_UNAUTHORIZED,
                )

            if (
                login_code.consumed_at is not None
                or login_code.expires_at <= now
            ):
                return Response(
                    {"error": "Invalid or expired OAuth code."},
                    status=status.HTTP_401_UNAUTHORIZED,
                )

            # Consume the code within the transaction so it cannot be reused.
            login_code.consumed_at = now
            login_code.save(update_fields=["consumed_at"])

            user = login_code.user

            if not user.is_active or user.is_blocked:
                return Response(
                    {"error": "This account is unavailable."},
                    status=status.HTTP_403_FORBIDDEN,
                )

            access_token = generate_access_token(user)
            refresh_token = generate_refresh_token(user)

            RefreshToken.objects.create(
                user=user,
                token=hash_refresh_token(refresh_token),
                expires_at=refresh_token_expires_at(),
            )

        return Response(
            {
                "access_token": access_token,
                "refresh_token": refresh_token,
            },
            status=status.HTTP_200_OK,
        )