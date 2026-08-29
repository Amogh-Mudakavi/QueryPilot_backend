from django.urls import path

from .views import (
    DocumentUploadAPIView,
    DocumentListAPIView,
    DocumentStatusAPIView,
    DocumentDeleteAPIView,
)


urlpatterns = [
    path(
        "upload/",
        DocumentUploadAPIView.as_view(),
        name="document-upload",
    ),

    path(
        "",
        DocumentListAPIView.as_view(),
        name="document-list",
    ),

    path(
        "<int:document_id>/status/",
        DocumentStatusAPIView.as_view(),
        name="document-status",
    ),

    path(
        "<int:document_id>/",
        DocumentDeleteAPIView.as_view(),
        name="document-delete",
    ),
]