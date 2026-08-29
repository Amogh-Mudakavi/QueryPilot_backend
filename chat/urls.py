"""
URL routes for QueryPilot chat.
"""

from django.urls import path

from chat.views import (
    ConversationDetailView,
    ConversationListCreateView,
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
        "messages/stream/",
        StreamMessageView.as_view(),
        name="stream-message",
    ),
]