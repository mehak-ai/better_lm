from django.urls import path
from .views import (
    UploadView,
    SessionListView,
    SessionDetailView,
    MessageListView,
    ChatView,
)

urlpatterns = [
    # POST  /api/upload/
    path("upload", UploadView.as_view(), name="upload_no_slash"),
    path("upload/", UploadView.as_view(), name="upload"),

    # GET   /api/sessions/
    path("sessions/", SessionListView.as_view(), name="session-list"),

    # GET   /api/sessions/<id>/
    # DELETE /api/sessions/<id>/
    path("sessions/<int:session_id>/", SessionDetailView.as_view(), name="session-detail"),

    # POST  /api/sessions/<id>/chat/
    path("sessions/<int:session_id>/chat", ChatView.as_view(), name="chat_no_slash"),
    path("sessions/<int:session_id>/chat/", ChatView.as_view(), name="chat"),

    # GET   /api/sessions/<id>/messages/
    path("sessions/<int:session_id>/messages", MessageListView.as_view(), name="messages_no_slash"),
    path("sessions/<int:session_id>/messages/", MessageListView.as_view(), name="messages"),
]
