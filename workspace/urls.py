"""Urls module for QueryPilot."""

from django.urls import path

from .views import (
    WorkspaceCreateAPIView,
    WorkspaceDetailAPIView,
    WorkspaceListAPIView,
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
