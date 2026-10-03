"""
HTTP views for QueryPilot chat.

Responsibilities:
    - Authenticate requests.
    - Validate request data.
    - Resolve conversation ownership.
    - Create conversations.
    - Read/update conversation settings.
    - Stream chat execution through Server-Sent Events.

The views do NOT:
    - perform retrieval
    - run LangGraph directly
    - call LLM providers directly
    - generate citations
    - implement agent logic
    - implement grounding logic
"""

from __future__ import annotations

import json

from django.http import StreamingHttpResponse
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.renderers import BaseRenderer
from rest_framework.response import Response
from rest_framework.views import APIView

from chat.models import Conversation, ConversationSettings
from chat.serializers import (
    ConversationSerializer,
    ConversationSettingsSerializer,
    CreateConversationSerializer,
    MessageSerializer,
    SendMessageSerializer,
)
from chat.services.conversation import (
    create_conversation,
    list_conversations,
)
from chat.services.execution import (
    execute_message_stream,
)
from utils.redis import check_chat_question_limit
from workspace.models import Workspace

CHAT_LIMIT_MESSAGE = (
    "Looks like I’ve hit the ceiling on the LLM credits available "
    "to this solo-developer project. I’ve exhausted the recovery "
    "options, so I’d rather be honest than invent an answer. "
    "QueryPilot is being built and funded by one developer using "
    "limited/free API capacity. If you see potential here, please "
    "consider hiring him, referring him, or connecting him with "
    "someone who can. Give him the opportunity and the resources, "
    "and this product can go much further."
)


class EventStreamRenderer(BaseRenderer):
    """
    Renderer used only for DRF content negotiation.

    The actual response is a StreamingHttpResponse, so DRF's
    renderer is not responsible for rendering the stream body.
    """

    media_type = "text/event-stream"
    format = "event-stream"
    charset = "utf-8"
    render_style = "binary"

    def render(
        self,
        data,
        accepted_media_type=None,
        renderer_context=None,
    ):
        return data


class ConversationListCreateView(APIView):
    """
    Create a new conversation or list conversations
    for a workspace.
    """

    permission_classes = [
        IsAuthenticated,
    ]

    def get(
        self,
        request,
    ):
        """
        Return conversations belonging to the requested workspace.
        """

        workspace_id = request.query_params.get("workspace_id")

        if not workspace_id:
            return Response(
                {
                    "detail": "workspace_id is required.",
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            workspace = Workspace.objects.get(
                id=workspace_id,
                owner=request.user,
            )
        except Workspace.DoesNotExist:
            return Response(
                {
                    "detail": "Workspace not found.",
                },
                status=status.HTTP_404_NOT_FOUND,
            )

        conversations = list_conversations(
            workspace=workspace,
        )

        serializer = ConversationSerializer(
            conversations,
            many=True,
        )

        return Response(
            serializer.data,
            status=status.HTTP_200_OK,
        )

    def post(
        self,
        request,
    ):
        """
        Create a new conversation in the authenticated user's workspace.
        """

        serializer = CreateConversationSerializer(
            data=request.data,
        )

        serializer.is_valid(
            raise_exception=True,
        )

        workspace_id = serializer.validated_data["workspace_id"]

        try:
            workspace = Workspace.objects.get(
                id=workspace_id,
                owner=request.user,
            )
        except Workspace.DoesNotExist:
            return Response(
                {
                    "detail": "Workspace not found.",
                },
                status=status.HTTP_404_NOT_FOUND,
            )

        conversation = create_conversation(
            workspace=workspace,
            user=request.user,
        )

        return Response(
            ConversationSerializer(
                conversation,
            ).data,
            status=status.HTTP_201_CREATED,
        )


class ConversationDetailView(APIView):
    """
    Return one conversation and its messages.
    """

    permission_classes = [
        IsAuthenticated,
    ]

    def get(
        self,
        request,
        conversation_id: int,
    ):
        """
        Return conversation metadata together with its messages.
        """

        try:
            conversation = Conversation.objects.get(
                id=conversation_id,
                created_by=request.user,
            )
        except Conversation.DoesNotExist:
            return Response(
                {
                    "detail": "Conversation not found.",
                },
                status=status.HTTP_404_NOT_FOUND,
            )

        conversation_data = ConversationSerializer(
            conversation,
        ).data

        messages = conversation.messages.all()

        conversation_data["messages"] = MessageSerializer(
            messages,
            many=True,
        ).data

        return Response(
            conversation_data,
            status=status.HTTP_200_OK,
        )


class ConversationSettingsView(APIView):
    """
    Read or update settings for one conversation.

    Settings are strictly scoped to the authenticated owner of the
    conversation.

    GET:
        Returns the current conversation settings.

    PATCH:
        Updates one or more settings.

    Older conversations that were created before the settings model
    existed are handled with get_or_create(), so they automatically
    receive the default settings.
    """

    permission_classes = [
        IsAuthenticated,
    ]

    def _get_conversation(
        self,
        request,
        conversation_id: int,
    ) -> Conversation | None:
        """
        Resolve a conversation belonging to the authenticated user.

        Returns None when the conversation does not exist or does not
        belong to the requesting user.
        """
        try:
            return Conversation.objects.get(
                id=conversation_id,
                created_by=request.user,
            )
        except Conversation.DoesNotExist:
            return None

    def get(
        self,
        request,
        conversation_id: int,
    ):
        """
        Return settings for an owned conversation.
        """

        conversation = self._get_conversation(
            request,
            conversation_id,
        )

        if conversation is None:
            return Response(
                {
                    "detail": "Conversation not found.",
                },
                status=status.HTTP_404_NOT_FOUND,
            )

        settings, _ = ConversationSettings.objects.get_or_create(
            conversation=conversation,
        )

        serializer = ConversationSettingsSerializer(
            settings,
        )

        return Response(
            serializer.data,
            status=status.HTTP_200_OK,
        )

    def patch(
        self,
        request,
        conversation_id: int,
    ):
        """
        Partially update settings for an owned conversation.
        """

        conversation = self._get_conversation(
            request,
            conversation_id,
        )

        if conversation is None:
            return Response(
                {
                    "detail": "Conversation not found.",
                },
                status=status.HTTP_404_NOT_FOUND,
            )

        settings, _ = ConversationSettings.objects.get_or_create(
            conversation=conversation,
        )

        serializer = ConversationSettingsSerializer(
            settings,
            data=request.data,
            partial=True,
        )

        serializer.is_valid(
            raise_exception=True,
        )

        serializer.save()

        return Response(
            serializer.data,
            status=status.HTTP_200_OK,
        )


class StreamMessageView(APIView):
    """Stream one QueryPilot message using Server-Sent Events.

    This is the canonical chat execution endpoint.

    User messages are persisted before execution begins. Successful assistant
    answers are persisted before the final `complete` event is emitted.
    """

    renderer_classes = [EventStreamRenderer]
    permission_classes = [
        IsAuthenticated,
    ]

    def post(
        self,
        request,
    ):
        """
        Validate the message request and return an SSE response.
        """

        serializer = SendMessageSerializer(
            data=request.data,
        )

        serializer.is_valid(
            raise_exception=True,
        )

        conversation_id = serializer.validated_data["conversation_id"]

        message = serializer.validated_data["message"]

        try:
            conversation = Conversation.objects.get(
                id=conversation_id,
                created_by=request.user,
            )
        except Conversation.DoesNotExist:
            return Response(
                {
                    "detail": "Conversation not found.",
                },
                status=status.HTTP_404_NOT_FOUND,
            )

        allowed, _ = check_chat_question_limit(
            user_id=request.user.id,
        )

        if not allowed:

            def limited_event_stream():
                payload = {
                    "message": CHAT_LIMIT_MESSAGE,
                    "data": {
                        "status": "rate_limited",
                        "retry_after_seconds": 7200,
                    },
                }

                yield (
                    "event: complete\n"
                    f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                )

            response = StreamingHttpResponse(
                limited_event_stream(),
                content_type="text/event-stream",
                status=200,
            )

            response["Cache-Control"] = "no-cache"
            response["X-Accel-Buffering"] = "no"

            return response

        def event_stream():
            """
            Convert ChatEvent objects into SSE wire format.

            Each event contains:

                event: <event-name>

                data: <JSON payload>
            """

            try:
                for event in execute_message_stream(
                    conversation_id=conversation.id,
                    user_message=message,
                ):
                    payload = {
                        "message": event.message,
                    }

                    if event.data is not None:
                        payload["data"] = event.data

                    yield (
                        f"event: {event.event}\n"
                        f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    )

            except Exception:
                error_payload = {
                    "message": ("We couldn't complete this request."),
                }

                yield (
                    "event: error\n"
                    f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"
                )

        response = StreamingHttpResponse(
            event_stream(),
            content_type="text/event-stream",
        )

        response["Cache-Control"] = "no-cache"
        response["X-Accel-Buffering"] = "no"

        return response
