"""Models module for QueryPilot."""

from django.db import models
from django.db.models.functions import Lower

from accounts.models import User


class Workspace(models.Model):
    owner = models.ForeignKey(User, on_delete=models.CASCADE, related_name="workspaces")

    title = models.CharField(max_length=255)

    created_at = models.DateTimeField(auto_now_add=True)

    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            # One workspace name per user, ignoring upper/lower case. Enforced by the
            # database, so two simultaneous requests (e.g. two browser tabs) cannot both win.
            models.UniqueConstraint(
                Lower("title"),
                "owner",
                name="uniq_workspace_title_per_owner_ci",
            ),
        ]

    def __str__(self):
        return self.title