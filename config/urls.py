from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),

    path("api/auth/", include("accounts.urls")),

    path("api/workspaces/", include("workspace.urls")),

    path("api/documents/", include("documents.urls")),
    path("api/chat/", include("chat.urls")),
]