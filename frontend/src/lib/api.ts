import { ChatSession, Message, Source } from "./types";

// Call the backend directly (works both server-side and in browser via CORS)
const BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

// ---- Sessions ----

export async function getSessions(): Promise<ChatSession[]> {
    const res = await fetch(`${BASE}/sessions/`, { cache: "no-store" });
    if (!res.ok) throw new Error("Failed to load sessions");
    return res.json();
}

export async function getSession(id: number): Promise<ChatSession> {
    const res = await fetch(`${BASE}/sessions/${id}/`, { cache: "no-store" });
    if (!res.ok) throw new Error("Session not found");
    return res.json();
}

export async function deleteSession(id: number): Promise<void> {
    await fetch(`${BASE}/sessions/${id}/`, { method: "DELETE" });
}

// ---- Create Session ----

export async function createSession(): Promise<ChatSession> {
    const res = await fetch(`${BASE}/sessions/create/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
    });
    if (!res.ok) throw new Error("Failed to create session");
    return res.json();
}

// ---- Upload ----

export async function uploadFiles(files: File[], sessionId?: number): Promise<ChatSession> {
    const form = new FormData();
    files.forEach((f) => form.append("files", f));
    if (sessionId) form.append("session_id", sessionId.toString());
    const res = await fetch(`${BASE}/upload/`, { method: "POST", body: form });
    if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error || "Upload failed");
    }
    return res.json();
}

// ---- Messages ----

export async function getMessages(sessionId: number): Promise<Message[]> {
    const res = await fetch(`${BASE}/sessions/${sessionId}/messages/`, {
        cache: "no-store",
    });
    if (!res.ok) throw new Error("Failed to load messages");
    return res.json();
}

// ---- Streaming Chat ----

export async function streamChat(
    sessionId: number,
    payload: {
        query: string;
        document_ids?: number[];
        page_start?: number | null;
        page_end?: number | null;
    },
    onToken: (token: string) => void,
    onDone: (sources: Source[], intent: string) => void,
    onError: (msg: string) => void
): Promise<void> {
    const res = await fetch(`${BASE}/sessions/${sessionId}/chat/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });

    if (!res.ok || !res.body) {
        onError("Request failed");
        return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            const raw = line.slice(6).trim();
            if (!raw) continue;
            try {
                const event = JSON.parse(raw);
                if (event.type === "token") onToken(event.content);
                else if (event.type === "done") onDone(event.sources, event.intent);
                else if (event.type === "error") onError(event.message);
            } catch {
                // skip malformed
            }
        }
    }
}

// ---- Voice Assistant ----

export async function getDeepgramToken(): Promise<string> {
    const res = await fetch(`${BASE}/voice/token/`);
    if (!res.ok) {
        console.error("Voice token error:", res.status, await res.text());
        throw new Error("Failed to get voice token");
    }
    const data = await res.json();
    return data.key;
}

export async function speakText(text: string): Promise<ArrayBuffer> {
    const res = await fetch(`${BASE}/voice/tts/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
    });
    if (!res.ok) throw new Error("TTS failed");
    return res.arrayBuffer();
}
