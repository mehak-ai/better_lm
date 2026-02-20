from django.db import models


class Document(models.Model):
    """Represents an uploaded research document."""

    title = models.CharField(max_length=512)
    authors = models.CharField(max_length=1024, blank=True)
    publication_date = models.DateField(null=True, blank=True)
    file = models.FileField(upload_to="documents/")
    file_type = models.CharField(max_length=10)  # pdf | docx | txt
    file_hash = models.CharField(max_length=64, unique=True, null=True, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    total_pages = models.IntegerField(default=0)

    class Meta:
        ordering = ["-uploaded_at"]

    def __str__(self):
        return self.title


class Chunk(models.Model):
    """A semantic chunk of a document with its vector embedding."""

    document = models.ForeignKey(
        Document, on_delete=models.CASCADE, related_name="chunks"
    )
    content = models.TextField()
    embedding = models.JSONField()  # Store as list of floats
    page_number = models.IntegerField(default=1)
    chunk_index = models.IntegerField(default=0)
    global_index = models.IntegerField(unique=True, null=True)

    class Meta:
        ordering = ["document_id", "chunk_index"]
        indexes = [
            models.Index(fields=["document_id", "page_number"]),
            models.Index(fields=["global_index"]),
        ]

    def __str__(self):
        return f"Chunk {self.chunk_index} | Doc {self.document_id} | p.{self.page_number}"


class ChatSession(models.Model):
    """A research session grouping one or more documents."""

    title = models.CharField(max_length=256)
    documents = models.ManyToManyField(Document, through="SessionDocument")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return self.title


class SessionDocument(models.Model):
    """Explicit through-table linking sessions to documents."""

    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE)
    document = models.ForeignKey(Document, on_delete=models.CASCADE)
    added_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("session", "document")


class Message(models.Model):
    """A chat message within a session."""

    ROLE_CHOICES = [("user", "User"), ("assistant", "Assistant")]

    session = models.ForeignKey(
        ChatSession, on_delete=models.CASCADE, related_name="messages"
    )
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    content = models.TextField()
    sources = models.JSONField(default=list, blank=True)  # structured citations
    intent = models.CharField(max_length=20, default="rag")  # rag|compare|mermaid
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]
