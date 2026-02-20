"use client";

import { useState } from "react";
import { Source } from "@/lib/types";

interface Props {
    sources: Source[];
}

export default function CitationList({ sources }: Props) {
    const [expanded, setExpanded] = useState<number | null>(null);

    if (!sources.length) return null;

    return (
        <div className="mt-3 pt-3 border-t" style={{ borderColor: "var(--border)" }}>
            <p className="text-[10px] font-bold uppercase tracking-widest mb-2"
                style={{ color: "var(--text-dim)" }}>
                Sources · {sources.length}
            </p>
            <div className="flex flex-wrap gap-1.5">
                {sources.map((s, i) => (
                    <div key={s.chunk_id} className="relative">
                        <button
                            className="citation-chip"
                            onClick={() => setExpanded(expanded === i ? null : i)}
                        >
                            <svg width="8" height="8" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                                <path d="M9 12h6M9 16h6M7 3H5a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2V5a2 2 0 00-2-2h-2" />
                                <rect x="7" y="1" width="10" height="4" rx="1" />
                            </svg>
                            p.{s.page_number} · {s.document_title.slice(0, 20)}{s.document_title.length > 20 ? "…" : ""}
                        </button>

                        {expanded === i && (
                            <div
                                className="absolute bottom-full left-0 mb-2 w-72 rounded-xl p-3 z-50 shadow-2xl"
                                style={{
                                    background: "var(--bg-elevated)",
                                    border: "1px solid var(--border-hover)",
                                }}
                            >
                                <div className="flex items-center justify-between mb-2">
                                    <p className="text-[10px] font-bold" style={{ color: "var(--accent)" }}>
                                        {s.document_title}
                                    </p>
                                    <span className="text-[9px] px-1.5 py-0.5 rounded-full font-bold"
                                        style={{ background: "var(--accent-dim)", color: "var(--accent)" }}>
                                        p.{s.page_number}
                                    </span>
                                </div>
                                <p className="text-xs leading-relaxed" style={{ color: "var(--text-primary)" }}>
                                    {s.excerpt}
                                </p>
                                <p className="text-[9px] mt-2 font-bold" style={{ color: "var(--text-dim)" }}>
                                    relevance: {Math.round(s.score * 100)}%
                                </p>
                            </div>
                        )}
                    </div>
                ))}
            </div>
        </div>
    );
}
