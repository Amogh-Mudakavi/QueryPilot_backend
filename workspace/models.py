"""Models module for QueryPilot."""

from django.db import models

from accounts.models import User


class Workspace(models.Model):
    owner = models.ForeignKey(User, on_delete=models.CASCADE, related_name="workspaces")

    title = models.CharField(max_length=255)

    created_at = models.DateTimeField(auto_now_add=True)

    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.title
