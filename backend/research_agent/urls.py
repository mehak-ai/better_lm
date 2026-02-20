from django.urls import path
from .views import (
    UploadView,
    CreateSessionView,
    SessionListView,
    SessionDetailView,
    MessageListView,
    ChatView,
    VoiceTokenView,
    VoiceTTSView,
)

urlpatterns = [
    # POST  /api/upload/
    path("upload", UploadView.as_view(), name="upload_no_slash"),
    path("upload/", UploadView.as_view(), name="upload"),

    # GET   /api/sessions/
    # POST  /api/sessions/           (create empty session)
    path("sessions/", SessionListView.as_view(), name="session-list"),
    path("sessions/create", CreateSessionView.as_view(), name="session-create-no-slash"),
    path("sessions/create/", CreateSessionView.as_view(), name="session-create"),

    # GET   /api/sessions/<id>/
    # DELETE /api/sessions/<id>/
    path("sessions/<int:session_id>/", SessionDetailView.as_view(), name="session-detail"),

    # POST  /api/sessions/<id>/chat/
    path("sessions/<int:session_id>/chat", ChatView.as_view(), name="chat_no_slash"),
    path("sessions/<int:session_id>/chat/", ChatView.as_view(), name="chat"),

    # GET   /api/sessions/<id>/messages/
    path("sessions/<int:session_id>/messages", MessageListView.as_view(), name="messages_no_slash"),
    path("sessions/<int:session_id>/messages/", MessageListView.as_view(), name="messages"),

    # Voice assistant
    path("voice/token", VoiceTokenView.as_view(), name="voice-token-no-slash"),
    path("voice/token/", VoiceTokenView.as_view(), name="voice-token"),
    path("voice/tts", VoiceTTSView.as_view(), name="voice-tts-no-slash"),
    path("voice/tts/", VoiceTTSView.as_view(), name="voice-tts"),
]
