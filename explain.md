# BetterLM: AI-Powered Research & Document Assistant

BetterLM is a sophisticated RAG (Retrieval-Augmented Generation) application designed for deep document analysis, comparison, and visualization. It allows users to upload multiple documents and interact with them using natural language, featuring a friendly voice assistant and automated diagram generation.

---

## 1. Technical Architecture & Workflow

The system follows a classic **RAG Architecture** with modern enhancements for multi-modal interaction.

### A. Document Ingestion Pipeline
1.  **Upload**: Files (PDF, DOCX, TXT) are uploaded via the frontend.
2.  **Extraction**: The backend uses `PDFSemanticProcessor` to identify structure (headings, pages).
3.  **Chunking**: Documents are split into semantic chunks to preserve context.
4.  **Embedding**: Chunks are converted into vector embeddings using an embedding model.
5.  **Storage**: Metadata and embeddings are stored in **PostgreSQL with pgvector**.

### B. Retrieval & Generation (Chat)
1.  **Query**: User asks a question.
2.  **Intent Classification**: The `IntentRouter` determines if the user wants a standard answer, a comparison, or a diagram (Mermaid).
3.  **Retrieval**: The `CustomRetriever` performs a similarity search in pgvector targeting the current session's documents.
4.  **Context Construction**: Retrieved chunks are formatted into a context string.
5.  **Memory**: The last 6 messages of conversation history are injected for follow-up query support.
6.  **LLM Execution**: The prompt is sent to the LLM (e.g., via Groq or OpenAI).
7.  **Streaming**: The response is streamed back to the frontend via SSE (Server-Sent Events).

---

## 2. Technical Stack

### **Frontend**
- **Framework**: Next.js 14+ (React)
- **Language**: TypeScript
- **Styling**: Vanilla CSS (Modern design patterns, glassmorphism, pulse animations)
- **Visualization**: Mermaid.js (for dynamic flowcharts and diagrams)
- **State Management**: React Hooks (useState, useEffect, useMemo, useCallback)
- **Communication**: Fetch API with SSE support for real-time text streaming.

### **Backend**
- **Framework**: Django & Django REST Framework (DRF)
- **AI Orchestration**: LangChain
- **Database**: PostgreSQL with `pgvector` for semantic search.
- **LLM Provider**: Groq (High-speed inference) / OpenAI.
- **Voice APIs**: 
  - **Deepgram**: Used for high-speed STT (Speech-to-Text) and TTS (Text-to-Speech) in the Voice Assistant.
  - **Web Speech API**: Used as a fallback for browser-native recognition.

---

## 3. Key Features

- **Semantic Document Analysis**: Understands the structure of PDFs, not just raw text.
- **Comparison Mode**: Specialized intent that explicitly looks for differences and agreements between multiple sources.
- **Mermaid.js Integration**: Automatically generates flowcharts or sequence diagrams based on document content.
- **Voice Assistant Overlay**: A ChatGPT-like full-screen voice experience with conversational memory.
- **Persistent Memory**: Remembers previous questions in a session for context-aware follow-ups.

---

## 4. Presentation Script & Demo Guide

Use this script to deliver a 5-10 minute presentation.

### **Part 1: The Hook (1 min)**
> "Today, we're overwhelmed with PDFs and documents. Traditional search only finds keywords; it doesn't understand context. I've built **BetterLM**, an assistant that doesn't just read your files—it researches them with you."

### **Part 2: Document Ingestion (1.5 mins)**
*Action: Upload 2-3 related PDFs (e.g., research papers or reports).*
> "I'll start by uploading these research papers. Behind the scenes, the system isn't just treating this as text. It's identifying headers, page numbers, and semantic sections. These are now indexed in a vector database for instant retrieval."

### **Part 3: Smart Querying & Memory (2 mins)**
*Action: Ask a specific question about one document. Then ask a follow-up like 'Can you explain that in simpler terms?'*
> "I can ask complex questions. Notice how it cites sources with page numbers. And because of the conversational memory we implemented, I can ask follow-up questions without repeating myself. It knows what 'that' refers to."

### **Part 4: Comparison & Visualization (2 mins)**
*Action: Use the comparison mode. Then ask 'Draw a flowchart explaining [System X]'.*
> "One of the most powerful features is **Intent Routing**. If I ask for a comparison, the AI specifically formats its logic to find discrepancies. If I ask for a diagram, it generates live Mermaid.js code that renders instantly on the screen."

### **Part 5: The Voice Assistant (1.5 mins)**
*Action: Click the microphone icon in the chat bar and speak your question. Let the AI respond with voice.*
> "Finally, for a more hands-free experience, we have the **Voice Assistant**. Using Deepgram's low-latency API, we can have a natural, spoken conversation with our data. It’s friendly, conversational, and highly responsive."

### **Part 6: Conclusion (1 min)**
> "BetterLM combines high-speed inference via Groq, semantic search via pgvector, and a premium React interface to turn static documents into interactive knowledge. Thank you!"

---

## 5. Development Workflow (How we built it)
1. **Infrastructure**: Set up Django and PostgreSQL with vector support.
2. **RAG Logic**: Implemented LangChain providers for document processing and LLM streaming.
3. **UI/UX**: Crafted a premium frontend focusing on "Rich Aesthetics"—using glassmorphism, animations, and a responsive toggle-based sidebar.
4. **Iterative Refinement**: Added memory, voice, and diagram support based on feature-driven development cycles.
