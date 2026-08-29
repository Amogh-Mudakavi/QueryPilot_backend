from django.urls import path

from .views import (
    WorkspaceCreateAPIView,
    WorkspaceListAPIView,
    WorkspaceDeleteAPIView,
)

urlpatterns = [
    path(
        "",
        WorkspaceListAPIView.as_view(),
    ),

    path(
        "create/",
        WorkspaceCreateAPIView.as_view(),
    ),

    path(
        "<int:workspace_id>/",
        WorkspaceDeleteAPIView.as_view(),
    ),
]