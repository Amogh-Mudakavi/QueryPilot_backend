from django.urls import path

from .views import (
    WorkspaceCreateAPIView,
    WorkspaceListAPIView,
    WorkspaceDetailAPIView,
    WorkspaceSettingsAPIView,
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
        WorkspaceDetailAPIView.as_view(),
    ),

    path(
        "<int:workspace_id>/settings/",
        WorkspaceSettingsAPIView.as_view(),
    ),
]