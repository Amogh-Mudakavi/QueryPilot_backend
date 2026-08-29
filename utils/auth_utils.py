import jwt
from accounts.models import User
from datetime import datetime, timedelta, timezone
from django.conf import settings
from rest_framework.response import Response
from rest_framework import status



def generate_access_token(user):

    payload = {
        "user_id": user.id,
        "email": user.email,
        "type": "access",
        "exp": datetime.now(timezone.utc) + timedelta(days = 1),
    }

    token = jwt.encode(
        payload,
        settings.JWT_SECRET_KEY,
        algorithm="HS256",
    )

    return token


def generate_refresh_token(user):

    payload = {
        "user_id": user.id,
        "type": "refresh",
        "exp": datetime.now(timezone.utc) + timedelta(days=7),
    }

    token = jwt.encode(
        payload,
        settings.JWT_SECRET_KEY,
        algorithm="HS256",
    )

    return token


def decode_token(token):

    payload = jwt.decode(
        token,
        settings.JWT_SECRET_KEY,
        algorithms=["HS256"],
    )

    return payload


def get_current_user(request):

    authorization = request.headers.get("Authorization")

    if not authorization:
        return None

    token = authorization.split(" ")[1]

    payload = decode_token(token)

    user_id = payload["user_id"]

    user = User.objects.get(id=user_id)

    if user.is_blocked:
        raise 

    return user

def admin_required(user):

    if user.role != "admin":

        return Response(
            {
                "error": "Admin access required"
            },
            status=status.HTTP_403_FORBIDDEN
        )

    return None