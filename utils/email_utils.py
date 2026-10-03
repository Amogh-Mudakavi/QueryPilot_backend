"""Email Utils module for QueryPilot."""

from datetime import timedelta

import resend
from django.conf import settings
from django.contrib.auth.hashers import check_password, make_password
from django.utils import timezone

from accounts.models import EmailOTP

resend.api_key = settings.RESEND_API_KEY


def generate_otp():
    import secrets

    return str(secrets.randbelow(900000) + 100000)


def hash_otp(otp):
    return make_password(otp)


def verify_otp(otp, otp_hash):
    return check_password(
        otp,
        otp_hash,
    )


def can_send_otp(email, ip_address=None):
    """
    Preserve the existing per-email resend cooldown.

    The IP argument exists so the caller can consistently pass the
    request IP. IP-based abuse protection is handled by
    check_hourly_limit().
    """

    latest_otp = EmailOTP.objects.filter(email=email).order_by("-created_at").first()

    if latest_otp is None:
        return True

    elapsed = timezone.now() - latest_otp.created_at

    return elapsed >= timedelta(seconds=settings.OTP_RESEND_COOLDOWN_SECONDS)


def check_hourly_limit(email, ip_address=None):
    """
    Enforce both per-email and per-IP OTP limits over the last hour.
    """

    one_hour_ago = timezone.now() - timedelta(hours=1)

    email_count = EmailOTP.objects.filter(
        created_at__gte=one_hour_ago,
        email=email,
    ).count()

    if email_count >= settings.OTP_HOURLY_LIMIT:
        return False

    if ip_address:
        ip_count = EmailOTP.objects.filter(
            created_at__gte=one_hour_ago,
            last_sent_ip=ip_address,
        ).count()

        if ip_count >= settings.OTP_IP_HOURLY_LIMIT:
            return False

    return True


def send_otp_email(email, otp):
    params = {
        "from": "HelpDesk AI <support@query-pilot.dev>",
        "to": [email],
        "subject": "Verify Your Email - HelpDesk AI",
        "html": f"""
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: auto;">
            <h2>Email Verification</h2>

            <p>Hello,</p>

            <p>Thank you for registering with <strong>HelpDesk AI</strong>.</p>

            <p>
                Please use the following One-Time Password (OTP)
                to verify your email address:
            </p>

            <h1 style="letter-spacing: 4px;">{otp}</h1>

            <p>
                This OTP is valid for
                <strong>{settings.OTP_LIFETIME_MINUTES} minutes</strong>.
            </p>

            <p>
                If you did not request this verification,
                you can safely ignore this email.
            </p>

            <hr>

            <p style="color: gray; font-size: 12px;">
                This is an automated email. Please do not reply.
            </p>
        </div>
        """,
    }

    return resend.Emails.send(params)
