"""Urls module for QueryPilot."""

from django.urls import path

from .oauth_views import (
    GoogleOAuthCallbackView,
    GoogleOAuthStartView,
    OAuthCodeExchangeView,
)
from .views import (
    ForgotPasswordRequestView,
    LoginView,
    LogoutView,
    RefreshView,
    RequestRegisterView,
    ResendRegistrationOTPView,
    ResetPasswordView,
    VerifyOTP,
    getme,
)

urlpatterns = [
    # Google OAuth
    path(
        "oauth/google/start/",
        GoogleOAuthStartView.as_view(),
    ),
    path(
        "oauth/google/callback/",
        GoogleOAuthCallbackView.as_view(),
    ),
    path(
        "oauth/exchange/",
        OAuthCodeExchangeView.as_view(),
    ),

    # Existing authentication endpoints
    path("register/", RequestRegisterView.as_view()),
    path("resend-register-otp/", ResendRegistrationOTPView.as_view()),
    path("verify-otp/", VerifyOTP.as_view()),
    path("forgot-password/", ForgotPasswordRequestView.as_view()),
    path("reset-password/", ResetPasswordView.as_view()),
    path("login/", LoginView.as_view()),
    path("refresh/", RefreshView.as_view()),
    path("logout/", LogoutView.as_view()),
    path("me/", getme.as_view()),
]