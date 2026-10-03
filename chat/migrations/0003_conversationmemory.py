"""Migrations 0003 Conversationmemory module for QueryPilot."""

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("chat", "0002_message_metadata"),
    ]

    operations = [
        migrations.CreateModel(
            name="ConversationMemory",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "summary",
                    models.TextField(
                        blank=True,
                        default="",
                        help_text="Rolling summary of older conversation context. This is context, not authoritative evidence.",
                    ),
                ),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "conversation",
                    models.OneToOneField(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="memory",
                        to="chat.conversation",
                    ),
                ),
                (
                    "summarized_until_message",
                    models.ForeignKey(
                        blank=True,
                        help_text="The latest message whose content is represented inside the rolling summary.",
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="+",
                        to="chat.message",
                    ),
                ),
            ],
        ),
    ]
