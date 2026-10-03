"""Urls module for QueryPilot."""

from django.urls import path

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
    path(
        "register/",
        RequestRegisterView.as_view(),
    ),
    path(
        "resend-register-otp/",
        ResendRegistrationOTPView.as_view(),
    ),
    path(
        "verify-otp/",
        VerifyOTP.as_view(),
    ),
    path(
        "forgot-password/",
        ForgotPasswordRequestView.as_view(),
    ),
    path(
        "reset-password/",
        ResetPasswordView.as_view(),
    ),
    path(
        "login/",
        LoginView.as_view(),
    ),
    path(
        "refresh/",
        RefreshView.as_view(),
    ),
    path(
        "logout/",
        LogoutView.as_view(),
    ),
    path(
        "me/",
        getme.as_view(),
    ),
]
