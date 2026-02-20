"use client";

import { useEffect, useRef } from "react";

interface Props {
    code: string;
}

export default function MermaidRenderer({ code }: Props) {
    const ref = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (!ref.current) return;

        const render = async () => {
            if (!code) return;

            // Clean code: remove markdown fences and find start of diagram
            let cleanCode = code.replace(/```mermaid/g, "").replace(/```/g, "").trim();
            const match = cleanCode.match(/^(flowchart|graph|sequenceDiagram|classDiagram|stateDiagram|erDiagram|gantt|pie|gitGraph|journey|timeline|mindmap|quadrantChart)/m);
            if (match && match.index !== undefined) {
                cleanCode = cleanCode.substring(match.index);
            }

            try {
                const mermaid = (await import("mermaid")).default;
                mermaid.initialize({
                    startOnLoad: false,
                    theme: "dark",
                    themeVariables: {
                        background: "#1c1c26",
                        primaryColor: "#7c6af7",
                        primaryTextColor: "#e8e8f0",
                        primaryBorderColor: "rgba(124,106,247,0.4)",
                        lineColor: "#6b6b80",
                        secondaryColor: "#222230",
                        tertiaryColor: "#16161e",
                        edgeLabelBackground: "#1c1c26",
                        fontFamily: "'Plus Jakarta Sans', sans-serif",
                    },
                    securityLevel: "loose",
                });

                const id = `mermaid-${Date.now()}`;
                // Render
                const { svg } = await mermaid.render(id, cleanCode);
                if (ref.current) {
                    ref.current.innerHTML = svg;
                }
            } catch (err) {
                console.error("Mermaid render error:", err);
                if (ref.current) {
                    ref.current.innerHTML = `<div class="p-4 text-xs text-red-400">
                        <p class="font-bold mb-1">Diagram Error</p>
                        <pre class="whitespace-pre-wrap">${err}</pre>
                    </div>`;
                }
            }
        };

        render();
    }, [code]);

    return (
        <div className="mermaid-output w-full overflow-auto rounded-xl p-4 my-2"
            style={{ background: "var(--bg-elevated)", border: "1px solid var(--border)" }}>
            <div className="flex items-center gap-2 mb-3">
                <div className="w-5 h-5 rounded-md flex items-center justify-center"
                    style={{ background: "var(--accent-dim)" }}>
                    <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
                        style={{ color: "var(--accent)" }}>
                        <rect x="3" y="3" width="18" height="18" rx="2" />
                        <path d="M3 9h18M9 21V9" />
                    </svg>
                </div>
                <span className="text-[10px] font-bold uppercase tracking-widest"
                    style={{ color: "var(--accent)" }}>
                    Diagram
                </span>
            </div>
            <div ref={ref} className="flex justify-center" />
        </div>
    );
}
