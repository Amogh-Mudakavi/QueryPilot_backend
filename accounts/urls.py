from django.urls import path
from .views import (
    RequestRegisterView,
    VerifyOTP,
    ForgotPasswordRequestView,
    ResetPasswordView,
    LoginView,
    getme,
    RefreshView,
    LogoutView,

)

urlpatterns = [
    # Auth
    path("register/", RequestRegisterView.as_view()),
    path("verify-otp/", VerifyOTP.as_view()),
    path("forgot-password/", ForgotPasswordRequestView.as_view()),
    path("reset-password/", ResetPasswordView.as_view()),
    path("login/", LoginView.as_view()),
    path("refresh/", RefreshView.as_view()),
    path("logout/", LogoutView.as_view()),
    path("me/", getme.as_view()),
]


