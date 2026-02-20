"use client";

import { useCallback, useState } from "react";
import { ChatSession } from "@/lib/types";
import { uploadFiles, deleteSession } from "@/lib/api";

interface Props {
    sessions: ChatSession[];
    activeId: number | null;
    onSelect: (s: ChatSession) => void;
    onNewSession: (s: ChatSession) => void;
    onDelete: (id: number) => void;
}

export default function Sidebar({
    sessions,
    activeId,
    onSelect,
    onNewSession,
    onDelete,
}: Props) {
    const [uploading, setUploading] = useState(false);
    const [dragOver, setDragOver] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const handleFiles = useCallback(
        async (files: File[]) => {
            if (!files.length) return;
            setUploading(true);
            setError(null);
            try {
                const session = await uploadFiles(files);
                onNewSession(session);
            } catch (e: unknown) {
                setError(e instanceof Error ? e.message : "Upload failed");
            } finally {
                setUploading(false);
            }
        },
        [onNewSession]
    );

    const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files) handleFiles(Array.from(e.target.files));
        e.target.value = "";
    };

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        setDragOver(false);
        if (e.dataTransfer.files) handleFiles(Array.from(e.dataTransfer.files));
    };

    const handleDelete = async (e: React.MouseEvent, id: number) => {
        e.stopPropagation();
        await deleteSession(id);
        onDelete(id);
    };

    const formatDate = (iso: string) => {
        const d = new Date(iso);
        const now = new Date();
        const diff = now.getTime() - d.getTime();
        if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
        if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
        if (diff < 604800000) return `${Math.floor(diff / 86400000)}d ago`;
        return d.toLocaleDateString();
    };

    return (
        <aside className="w-72 flex-shrink-0 flex flex-col h-full"
            style={{ background: "var(--bg-secondary)", borderRight: "1px solid var(--border)" }}>

            {/* Logo */}
            <div className="p-6 pb-4">
                <div className="flex items-center gap-3 mb-6">
                    <div className="w-8 h-8 rounded-xl flex items-center justify-center"
                        style={{ background: "var(--accent)" }}>
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5">
                            <path d="M12 2L2 7l10 5 10-5-10-5z" />
                            <path d="M2 17l10 5 10-5" />
                            <path d="M2 12l10 5 10-5" />
                        </svg>
                    </div>
                    <span className="font-bold text-sm tracking-tight" style={{ color: "var(--text-primary)" }}>
                        Research Agent
                    </span>
                </div>

                {/* Upload Button */}
                <label
                    className={`flex items-center justify-center gap-2 w-full py-3 rounded-xl text-xs font-bold uppercase tracking-widest cursor-pointer transition-all ${uploading ? "opacity-60 cursor-wait" : "hover:opacity-90"
                        } ${dragOver ? "drop-zone-active" : ""}`}
                    style={{ background: "var(--accent)", color: "white" }}
                    onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                    onDragLeave={() => setDragOver(false)}
                    onDrop={handleDrop}
                >
                    <input
                        type="file"
                        multiple
                        accept=".pdf,.docx,.txt"
                        className="hidden"
                        onChange={handleInputChange}
                        disabled={uploading}
                    />
                    {uploading ? (
                        <>
                            <svg className="animate-spin w-3 h-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
                            </svg>
                            Processing…
                        </>
                    ) : (
                        <>
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                                <path d="M12 5v14M5 12l7-7 7 7" />
                            </svg>
                            Upload Document
                        </>
                    )}
                </label>

                {error && (
                    <p className="mt-2 text-xs text-red-400 text-center">{error}</p>
                )}
            </div>

            {/* Sessions */}
            <div className="flex-1 overflow-y-auto px-3 pb-6">
                <p className="text-[10px] font-bold uppercase tracking-[0.2em] mb-3 px-3"
                    style={{ color: "var(--text-dim)" }}>
                    Recent Sessions
                </p>

                {sessions.length === 0 ? (
                    <div className="text-center py-8 px-4">
                        <div className="w-10 h-10 rounded-xl mx-auto mb-3 flex items-center justify-center"
                            style={{ background: "var(--bg-elevated)" }}>
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"
                                style={{ color: "var(--text-dim)" }}>
                                <path d="M9 12h6M9 16h6M7 3H5a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2V5a2 2 0 00-2-2h-2" />
                                <rect x="7" y="1" width="10" height="4" rx="1" />
                            </svg>
                        </div>
                        <p className="text-xs" style={{ color: "var(--text-muted)" }}>
                            Upload a document to start
                        </p>
                    </div>
                ) : (
                    <div className="space-y-1">
                        {sessions.map((s) => (
                            <div
                                key={s.id}
                                onClick={() => onSelect(s)}
                                className="group relative flex items-start gap-3 p-3 rounded-xl cursor-pointer transition-all"
                                style={{
                                    background: activeId === s.id ? "var(--accent-dim)" : "transparent",
                                    border: `1px solid ${activeId === s.id ? "rgba(124,106,247,0.3)" : "transparent"}`,
                                }}
                            >
                                <div className="w-6 h-6 rounded-lg flex-shrink-0 flex items-center justify-center mt-0.5"
                                    style={{ background: activeId === s.id ? "var(--accent)" : "var(--bg-elevated)" }}>
                                    <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                                        <path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01" />
                                    </svg>
                                </div>
                                <div className="flex-1 min-w-0">
                                    <p className="text-xs font-semibold truncate leading-tight mb-0.5"
                                        style={{ color: "var(--text-primary)" }}>
                                        {s.title}
                                    </p>
                                    <p className="text-[10px]" style={{ color: "var(--text-muted)" }}>
                                        {s.documents.length} doc{s.documents.length !== 1 ? "s" : ""} · {formatDate(s.created_at)}
                                    </p>
                                </div>
                                <button
                                    onClick={(e) => handleDelete(e, s.id)}
                                    className="opacity-0 group-hover:opacity-100 p-1 rounded-lg transition-all hover:bg-red-500/20"
                                    style={{ color: "var(--text-muted)" }}
                                >
                                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                        <path d="M18 6L6 18M6 6l12 12" />
                                    </svg>
                                </button>
                            </div>
                        ))}
                    </div>
                )}
            </div>

            {/* Footer */}
            <div className="p-4 border-t" style={{ borderColor: "var(--border)" }}>
                <div className="flex items-center gap-2 p-2 rounded-xl"
                    style={{ background: "var(--bg-elevated)" }}>
                    <div className="w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold"
                        style={{ background: "var(--accent)", color: "white" }}>
                        R
                    </div>
                    <div>
                        <p className="text-xs font-semibold" style={{ color: "var(--text-primary)" }}>Researcher</p>
                        <p className="text-[10px]" style={{ color: "var(--text-muted)" }}>Free tier</p>
                    </div>
                </div>
            </div>
        </aside>
    );
}
