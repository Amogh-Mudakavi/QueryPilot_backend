from django.conf import settings
import secrets 
from accounts.models import EmailOTP
from datetime import timedelta
from django.utils import timezone
import resend


resend.api_key = settings.RESEND_API_KEY
def generate_otp():

    return str(
        secrets.randbelow(900000) + 100000
    )


def can_send_otp(email):

    latest_otp = EmailOTP.objects.filter(email = email).order_by('-created_at').first()


    if latest_otp is None:
        return True
    
    current_time = timezone.now()

    time_difference = current_time - latest_otp.created_at

    if time_difference < timedelta(seconds=60):
        return False

    return True

def check_hourly_limit(email):

    one_hr_ago = timezone.now() - timedelta(hours=1)

    otp_count = EmailOTP.objects.filter(created_at__gte = one_hr_ago , email = email).count()

    if otp_count >=3:
        return False

    return True
    

def send_otp_email(email , otp):

    params = {
    "from": "HelpDesk AI <support@query-pilot.dev>",
    "to": [email],
    "subject": "Verify Your Email - HelpDesk AI",
    "html": f"""
    <div style="font-family: Arial, sans-serif; max-width: 600px; margin: auto;">
        <h2>Email Verification</h2>

        <p>Hello,</p>

        <p>Thank you for registering with <strong>HelpDesk AI</strong>.</p>

        <p>Please use the following One-Time Password (OTP) to verify your email address:</p>

        <h1 style="letter-spacing: 4px;">{otp}</h1>

        <p>This OTP is valid for <strong>10 minutes</strong>.</p>

        <p>If you did not request this verification, you can safely ignore this email.</p>

        <hr>

        <p style="color: gray; font-size: 12px;">
            This is an automated email. Please do not reply.
        </p>
    </div>
    """
}

    return resend.Emails.send(params)

