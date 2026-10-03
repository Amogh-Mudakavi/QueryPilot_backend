"""
URL routes for QueryPilot chat.
"""

from django.urls import path

from chat.views import (
    ConversationDetailView,
    ConversationListCreateView,
    ConversationSettingsView,
    StreamMessageView,
)

urlpatterns = [
    path(
        "conversations/",
        ConversationListCreateView.as_view(),
        name="conversation-list-create",
    ),
    path(
        "conversations/<int:conversation_id>/",
        ConversationDetailView.as_view(),
        name="conversation-detail",
    ),
    path(
        "conversations/<int:conversation_id>/settings/",
        ConversationSettingsView.as_view(),
        name="conversation-settings",
    ),
    path(
        "messages/stream/",
        StreamMessageView.as_view(),
        name="stream-message",
    ),
]
