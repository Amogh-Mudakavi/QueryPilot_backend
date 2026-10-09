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
    from html import escape

    lifetime = settings.OTP_LIFETIME_MINUTES
    safe_otp = escape(str(otp))

    params = {
        "from": "QueryPilot <support@query-pilot.dev>",
        "to": [email],
        "subject": "Your QueryPilot verification code",
        "html": f"""\
<!DOCTYPE html>
<html lang="en">
<body style="margin:0;padding:0;background:#F4F1EA;
    font-family:Geist,-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;
    color:#171717;">

    <table role="presentation" width="100%" cellspacing="0" cellpadding="0"
           style="background:#F4F1EA;padding:40px 16px;">
        <tr>
            <td align="center">
                <table role="presentation" width="100%" cellspacing="0"
                       cellpadding="0" style="max-width:480px;background:#FFFFFF;
                       border:1px solid #E8E4DC;border-radius:10px;">

                    <tr>
                        <td style="padding:30px 32px 24px;">
                            <div style="font-size:22px;font-weight:650;
                                        letter-spacing:-0.8px;">
                                QueryPilot
                            </div>
                        </td>
                    </tr>

                    <tr>
                        <td style="padding:0 32px;">
                            <div style="height:1px;background:#ECEAE5;"></div>
                        </td>
                    </tr>

                    <tr>
                        <td style="padding:30px 32px 12px;">
                            <p style="margin:0 0 12px;font-size:11px;
                                      font-weight:600;letter-spacing:1.5px;
                                      color:#737373;">
                                ACCOUNT VERIFICATION
                            </p>

                            <h1 style="margin:0 0 14px;font-size:25px;
                                       line-height:1.3;font-weight:650;
                                       letter-spacing:-0.8px;">
                                Verify your email address
                            </h1>

                            <p style="margin:0;font-size:14px;line-height:1.8;
                                      color:#666666;">
                                Thanks for signing up for QueryPilot.
                                Enter the verification code below to complete
                                your registration.
                            </p>
                        </td>
                    </tr>

                    <tr>
                        <td style="padding:22px 32px;">
                            <table role="presentation" width="100%"
                                   cellspacing="0" cellpadding="0"
                                   style="background:#F8F6F1;
                                   border:1px solid #EAE6DE;border-radius:8px;">
                                <tr>
                                    <td align="center" style="padding:24px 12px;">
                                        <p style="margin:0 0 12px;font-size:10px;
                                                  font-weight:600;
                                                  letter-spacing:1.8px;
                                                  color:#737373;">
                                            VERIFICATION CODE
                                        </p>

                                        <p style="margin:0;font-family:
                                            'Geist Mono',ui-monospace,
                                            'SFMono-Regular',Consolas,monospace;
                                            font-size:32px;font-weight:600;
                                            letter-spacing:7px;color:#171717;
                                            overflow-wrap:anywhere;">
                                            {safe_otp}
                                        </p>

                                        <p style="margin:14px 0 0;font-size:12px;
                                                  color:#666666;">
                                            Expires in
                                            <strong>{lifetime} minutes</strong>
                                        </p>
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>

                    <tr>
                        <td style="padding:0 32px 30px;">
                            <p style="margin:0 0 12px;font-size:13px;
                                      line-height:1.8;color:#525252;">
                                <strong style="color:#171717;">
                                    Keep your account secure.
                                </strong>
                                Never share this code with anyone.
                            </p>

                            <p style="margin:0;font-size:13px;line-height:1.8;
                                      color:#737373;">
                                Didn't request this code? You can safely ignore
                                this email.
                            </p>
                        </td>
                    </tr>

                    <tr>
                        <td style="padding:20px 24px;border-top:1px solid #ECEAE5;
                                   text-align:center;">
                            <p style="margin:0 0 6px;font-size:11px;
                                      color:#737373;">
                                Secure access to your knowledge.
                            </p>
                            <p style="margin:0;font-size:11px;color:#999999;">
                                Automated email. Please do not reply.
                            </p>
                        </td>
                    </tr>

                </table>
            </td>
        </tr>
    </table>
</body>
</html>
""",
        "text": (
            "QueryPilot — Verify your email address\n\n"
            "Enter this verification code to complete your registration:\n\n"
            f"{otp}\n\n"
            f"This code expires in {lifetime} minutes.\n\n"
            "Never share this code with anyone. If you did not request it, "
            "you can safely ignore this email.\n\n"
            "QueryPilot"
        ),
    }

    return resend.Emails.send(params)