from django.apps import AppConfig


class ResearchAgentConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "research_agent"

    def ready(self):
        # Pre-warm embedding model on startup so first request is fast
        try:
            from .services import EmbeddingService
            EmbeddingService.get_model()
        except Exception:
            pass
