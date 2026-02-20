"use client";

import { useState, useEffect } from "react";
import Sidebar from "@/components/Sidebar";
import Chat from "@/components/Chat";
import { ChatSession } from "@/lib/types";
import { getSessions } from "@/lib/api";

export default function Home() {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSession, setActiveSession] = useState<ChatSession | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getSessions()
      .then((s) => {
        setSessions(s);
        if (s.length > 0) setActiveSession(s[0]);
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  const handleNewSession = (s: ChatSession) => {
    setSessions((prev) => [s, ...prev]);
    setActiveSession(s);
  };

  const handleDelete = (id: number) => {
    setSessions((prev) => prev.filter((s) => s.id !== id));
    if (activeSession?.id === id) {
      const remaining = sessions.filter((s) => s.id !== id);
      setActiveSession(remaining[0] ?? null);
    }
  };

  return (
    <div className="flex h-screen overflow-hidden" style={{ background: "var(--bg-primary)" }}>
      <Sidebar
        sessions={sessions}
        activeId={activeSession?.id ?? null}
        onSelect={setActiveSession}
        onNewSession={handleNewSession}
        onDelete={handleDelete}
      />

      <main className="flex-1 overflow-hidden">
        {loading ? (
          <div className="flex items-center justify-center h-full">
            <div className="flex flex-col items-center gap-4">
              <div className="w-10 h-10 rounded-xl flex items-center justify-center"
                style={{ background: "var(--accent)" }}>
                <svg className="animate-spin w-5 h-5 text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
                </svg>
              </div>
              <p className="text-xs font-bold uppercase tracking-widest"
                style={{ color: "var(--text-muted)" }}>Loading…</p>
            </div>
          </div>
        ) : activeSession ? (
          <Chat key={activeSession.id} session={activeSession} />
        ) : (
          /* Welcome / empty state */
          <div className="flex flex-col items-center justify-center h-full px-8">
            <div className="max-w-md text-center">
              <div
                className="w-20 h-20 rounded-3xl flex items-center justify-center mx-auto mb-6"
                style={{ background: "var(--accent)", boxShadow: "0 20px 60px rgba(124,106,247,0.35)" }}
              >
                <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2">
                  <path d="M12 2L2 7l10 5 10-5-10-5z" />
                  <path d="M2 17l10 5 10-5" />
                  <path d="M2 12l10 5 10-5" />
                </svg>
              </div>
              <h1 className="text-2xl font-bold mb-3" style={{ color: "var(--text-primary)" }}>
                Smart Research Agent
              </h1>
              <p className="text-sm leading-relaxed mb-8" style={{ color: "var(--text-muted)" }}>
                Create a new session and upload your research documents to start
                asking questions. Powered by pgvector semantic search and Groq AI
                with streaming responses.
              </p>

              <div className="grid grid-cols-3 gap-4">
                {[
                  { icon: "🔍", title: "Neural Search", desc: "pgvector cosine similarity" },
                  { icon: "⚡", title: "Streaming", desc: "Real-time AI responses" },
                  { icon: "📊", title: "Diagrams", desc: "Mermaid.js generation" },
                ].map((f) => (
                  <div
                    key={f.title}
                    className="rounded-2xl p-4 text-center"
                    style={{ background: "var(--bg-surface)", border: "1px solid var(--border)" }}
                  >
                    <div className="text-2xl mb-2">{f.icon}</div>
                    <p className="text-xs font-bold mb-1" style={{ color: "var(--text-primary)" }}>
                      {f.title}
                    </p>
                    <p className="text-[10px]" style={{ color: "var(--text-muted)" }}>
                      {f.desc}
                    </p>
                  </div>
                ))}
              </div>

              <p className="text-xs mt-8" style={{ color: "var(--text-dim)" }}>
                ← Click &quot;New Session&quot; in the sidebar to begin
              </p>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
