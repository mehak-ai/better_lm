"use client";

import {
    useState,
    useEffect,
    useRef,
    useCallback,
    useMemo,
} from "react";
import dynamic from "next/dynamic";
import { ChatSession, Document, Message, Source } from "@/lib/types";
import { getMessages, streamChat, uploadFiles } from "@/lib/api";
import CitationList from "./CitationList";
import VoiceAssistant from "./VoiceAssistant";

const MermaidRenderer = dynamic(() => import("./MermaidRenderer"), {
    ssr: false,
});

interface Props {
    session: ChatSession;
}

// Intent badge
const intentMeta = {
    rag: { label: "RAG", color: "#7c6af7" },
    compare: { label: "Compare", color: "#f7916a" },
    mermaid: { label: "Diagram", color: "#4ade80" },
};

function IntentBadge({ intent }: { intent: string }) {
    const meta = intentMeta[intent as keyof typeof intentMeta] ?? intentMeta.rag;
    return (
        <span
            className="text-[9px] font-bold uppercase tracking-widest px-2 py-0.5 rounded-full"
            style={{ background: `${meta.color}22`, color: meta.color, border: `1px solid ${meta.color}44` }}
        >
            {meta.label}
        </span>
    );
}

// Detect mermaid content
function isMermaidContent(content: string): boolean {
    return /^(flowchart|graph|sequenceDiagram|classDiagram|stateDiagram|erDiagram|gantt|pie|gitGraph|journey|timeline|mindmap|quadrantChart)/m.test(
        content.trim()
    );
}

// Render message body — mermaid or prose
function MessageBody({
    content,
    intent,
    streaming,
}: {
    content: string;
    intent: string;
    streaming?: boolean;
}) {
    if (intent === "mermaid" && !streaming && isMermaidContent(content)) {
        return <MermaidRenderer code={content.trim()} />;
    }
    return (
        <p
            className={`text-sm leading-relaxed whitespace-pre-wrap ${streaming ? "streaming-cursor" : ""}`}
            style={{ color: "var(--text-primary)" }}
        >
            {content}
        </p>
    );
}

export default function Chat({ session }: Props) {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState("");
    const [loading, setLoading] = useState(false);
    // Initialize sessionDocs from prop
    const [sessionDocs, setSessionDocs] = useState<Document[]>(session.documents);
    const [selectedDocs, setSelectedDocs] = useState<number[]>([]);
    const [pageStart, setPageStart] = useState<string>("");
    const [pageEnd, setPageEnd] = useState<string>("");
    const [showDocPanel, setShowDocPanel] = useState(false);
    const [isListening, setIsListening] = useState(false);
    const [speakingMsgIdx, setSpeakingMsgIdx] = useState<number | null>(null);
    const [showVoice, setShowVoice] = useState(false);
    const bottomRef = useRef<HTMLDivElement>(null);
    const textareaRef = useRef<HTMLTextAreaElement>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const recognitionRef = useRef<any>(null);

    // Load history
    useEffect(() => {
        setMessages([]);
        setSessionDocs(session.documents);
        setSelectedDocs(session.documents.map((d) => d.id));
        getMessages(session.id)
            .then(setMessages)
            .catch(console.error);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [session.id]);

    // Scroll to bottom on new messages
    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages]);

    // Auto-resize textarea
    useEffect(() => {
        if (textareaRef.current) {
            textareaRef.current.style.height = "auto";
            textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 160)}px`;
        }
    }, [input]);

    // Show doc panel momentarily on new uploads
    const prevDocCount = useRef(session.documents.length);
    useEffect(() => {
        if (sessionDocs.length > prevDocCount.current) {
            setShowDocPanel(true);
            const t = setTimeout(() => setShowDocPanel(false), 2500);
            return () => clearTimeout(t);
        }
        prevDocCount.current = sessionDocs.length;
    }, [sessionDocs]);

    // ---- Voice: Speech-to-Text ----
    const toggleListening = useCallback(() => {
        // Check browser support
        const SpeechRecognition =
            (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
        if (!SpeechRecognition) {
            alert("Speech recognition is not supported in this browser. Try Chrome.");
            return;
        }

        if (isListening && recognitionRef.current) {
            recognitionRef.current.stop();
            setIsListening(false);
            return;
        }

        const recognition = new SpeechRecognition();
        recognition.lang = "en-US";
        recognition.interimResults = true;
        recognition.continuous = true;
        recognitionRef.current = recognition;

        let finalTranscript = input; // Append to existing input

        recognition.onresult = (event: any) => {
            let interim = "";
            for (let i = event.resultIndex; i < event.results.length; i++) {
                const transcript = event.results[i][0].transcript;
                if (event.results[i].isFinal) {
                    finalTranscript += (finalTranscript ? " " : "") + transcript;
                } else {
                    interim = transcript;
                }
            }
            setInput(finalTranscript + (interim ? " " + interim : ""));
        };

        recognition.onerror = (event: any) => {
            console.error("Speech recognition error:", event.error);
            setIsListening(false);
        };

        recognition.onend = () => {
            setIsListening(false);
            recognitionRef.current = null;
        };

        recognition.start();
        setIsListening(true);
    }, [isListening, input]);

    // Cleanup speech recognition on unmount
    useEffect(() => {
        return () => {
            if (recognitionRef.current) {
                recognitionRef.current.stop();
            }
            window.speechSynthesis?.cancel();
        };
    }, []);

    // ---- Voice: Text-to-Speech ----
    const handleSpeak = useCallback((text: string, msgIdx: number) => {
        const synth = window.speechSynthesis;
        if (!synth) {
            alert("Text-to-speech is not supported in this browser.");
            return;
        }

        // If already speaking this message, stop
        if (speakingMsgIdx === msgIdx) {
            synth.cancel();
            setSpeakingMsgIdx(null);
            return;
        }

        // Cancel any current speech
        synth.cancel();

        // Clean text (remove mermaid code, markdown artifacts)
        const cleanText = text
            .replace(/```[\s\S]*?```/g, "")
            .replace(/[#*_`]/g, "")
            .trim();

        if (!cleanText) return;

        const utterance = new SpeechSynthesisUtterance(cleanText);
        utterance.rate = 1.0;
        utterance.pitch = 1.0;
        utterance.onend = () => setSpeakingMsgIdx(null);
        utterance.onerror = () => setSpeakingMsgIdx(null);

        setSpeakingMsgIdx(msgIdx);
        synth.speak(utterance);
    }, [speakingMsgIdx]);

    const toggleDoc = useCallback((id: number) => {
        setSelectedDocs((prev) =>
            prev.includes(id) ? prev.filter((d) => d !== id) : [...prev, id]
        );
    }, []);

    const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        if (!e.target.files?.length) return;

        try {
            setLoading(true);
            const updatedSession = await uploadFiles(
                Array.from(e.target.files),
                session.id
            );

            // Update local state
            setSessionDocs(updatedSession.documents);

            // Add new docs to selected
            const newDocIds = updatedSession.documents.map((d: any) => d.id);
            setSelectedDocs(newDocIds);

        } catch (err) {
            console.error(err);
            alert("Failed to upload files");
        } finally {
            setLoading(false);
            if (fileInputRef.current) fileInputRef.current.value = "";
        }
    };

    const handleSend = useCallback(async () => {
        const q = input.trim();
        if (!q || loading) return;

        const userMsg: Message = {
            role: "user",
            content: q,
            sources: [],
            intent: "rag",
        };
        setMessages((prev) => [...prev, userMsg]);
        setInput("");
        setLoading(true);

        // Optimistic streaming assistant placeholder
        const assistantMsg: Message = {
            role: "assistant",
            content: "",
            sources: [],
            intent: "rag",
            streaming: true,
        };
        setMessages((prev) => [...prev, assistantMsg]);

        let finalContent = "";

        await streamChat(
            session.id,
            {
                query: q,
                document_ids: selectedDocs,
                page_start: pageStart ? parseInt(pageStart) : null,
                page_end: pageEnd ? parseInt(pageEnd) : null,
            },
            // onToken
            (token) => {
                finalContent += token;
                setMessages((prev) => {
                    const updated = [...prev];
                    const last = { ...updated[updated.length - 1] };
                    last.content = finalContent;
                    updated[updated.length - 1] = last;
                    return updated;
                });
            },
            // onDone
            (sources: Source[], intent: string) => {
                setMessages((prev) => {
                    const updated = [...prev];
                    const last = { ...updated[updated.length - 1] };
                    last.sources = sources;
                    last.intent = intent as Message["intent"];
                    last.streaming = false;
                    updated[updated.length - 1] = last;
                    return updated;
                });
                setLoading(false);
            },
            // onError
            (err) => {
                setMessages((prev) => {
                    const updated = [...prev];
                    const last = { ...updated[updated.length - 1] };
                    last.content = `Error: ${err}`;
                    last.streaming = false;
                    updated[updated.length - 1] = last;
                    return updated;
                });
                setLoading(false);
            }
        );
    }, [input, loading, session.id, selectedDocs, pageStart, pageEnd]);

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    const allSelected = useMemo(
        () => sessionDocs.every((d) => selectedDocs.includes(d.id)),
        [sessionDocs, selectedDocs]
    );

    return (
        <div className="flex flex-col h-full">
            {/* Header */}
            <header
                className="flex items-center justify-between px-6 py-4 border-b"
                style={{ borderColor: "var(--border)", background: "var(--bg-secondary)" }}
            >
                <div>
                    <h1 className="font-bold text-sm truncate max-w-md" style={{ color: "var(--text-primary)" }}>
                        {session.title}
                    </h1>
                    <p className="text-[10px] mt-0.5" style={{ color: "var(--text-muted)" }}>
                        {sessionDocs.length} document{sessionDocs.length !== 1 ? "s" : ""} · {selectedDocs.length} selected
                    </p>
                </div>

                <div className="flex items-center gap-2">
                    {/* Voice Assistant Button */}
                    <button
                        onClick={() => setShowVoice(true)}
                        className="flex items-center gap-2 px-3 py-2 rounded-xl text-xs font-semibold transition-all"
                        style={{
                            background: "var(--bg-elevated)",
                            color: "var(--text-muted)",
                            border: "1px solid var(--border)",
                        }}
                        title="Voice conversation"
                    >
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M12 1a3 3 0 00-3 3v8a3 3 0 006 0V4a3 3 0 00-3-3z" />
                            <path d="M19 10v2a7 7 0 01-14 0v-2" />
                        </svg>
                        Voice
                    </button>

                    <button
                        onClick={() => setShowDocPanel(!showDocPanel)}
                        className="flex items-center gap-2 px-3 py-2 rounded-xl text-xs font-semibold transition-all"
                        style={{
                            background: showDocPanel ? "var(--accent-dim)" : "var(--bg-elevated)",
                            color: showDocPanel ? "var(--accent)" : "var(--text-muted)",
                            border: `1px solid ${showDocPanel ? "rgba(124,106,247,0.3)" : "var(--border)"}`,
                        }}
                    >
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M9 12h6M9 16h6M7 3H5a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2V5a2 2 0 00-2-2h-2" />
                            <rect x="7" y="1" width="10" height="4" rx="1" />
                        </svg>
                        Documents
                    </button>
                </div>
            </header>

            <div className="flex flex-1 overflow-hidden">
                {/* Messages */}
                <div className="flex-1 flex flex-col overflow-hidden">
                    <div className="flex-1 overflow-y-auto px-6 py-6 space-y-6">
                        {messages.length === 0 && (
                            <div className="flex flex-col items-center justify-center h-full text-center py-16">
                                {sessionDocs.length === 0 ? (
                                    /* No documents yet — prompt upload */
                                    <>
                                        <div
                                            className="w-16 h-16 rounded-2xl flex items-center justify-center mb-4"
                                            style={{ background: "var(--bg-elevated)", border: "1px solid var(--border)" }}
                                        >
                                            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"
                                                style={{ color: "var(--accent)" }}>
                                                <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
                                                <path d="M14 2v6h6" />
                                                <path d="M12 18v-6" />
                                                <path d="M9 15l3-3 3 3" />
                                            </svg>
                                        </div>
                                        <h2 className="font-bold text-base mb-2" style={{ color: "var(--text-primary)" }}>
                                            Upload documents to get started
                                        </h2>
                                        <p className="text-xs max-w-xs mb-6" style={{ color: "var(--text-muted)" }}>
                                            Use the <strong>+</strong> button below to upload PDFs, DOCX, or TXT files to this session.
                                        </p>
                                        <button
                                            onClick={() => fileInputRef.current?.click()}
                                            className="flex items-center gap-2 px-5 py-3 rounded-xl text-xs font-bold uppercase tracking-widest transition-all hover:opacity-90"
                                            style={{ background: "var(--accent)", color: "white" }}
                                        >
                                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                                                <path d="M12 5v14M5 12h14" />
                                            </svg>
                                            Upload Documents
                                        </button>
                                    </>
                                ) : (
                                    /* Has documents — ready to chat */
                                    <>
                                        <div
                                            className="w-16 h-16 rounded-2xl flex items-center justify-center mb-4"
                                            style={{ background: "var(--bg-elevated)", border: "1px solid var(--border)" }}
                                        >
                                            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"
                                                style={{ color: "var(--accent)" }}>
                                                <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z" />
                                            </svg>
                                        </div>
                                        <h2 className="font-bold text-base mb-2" style={{ color: "var(--text-primary)" }}>
                                            Ready to research
                                        </h2>
                                        <p className="text-xs max-w-xs" style={{ color: "var(--text-muted)" }}>
                                            Ask questions, compare documents, or request a diagram. Try: <em>&quot;draw a flow diagram of…&quot;</em>
                                        </p>
                                        <div className="flex gap-2 mt-6 flex-wrap justify-center">
                                            {[
                                                "Summarize the key findings",
                                                "Compare both documents",
                                                "Draw a diagram of the architecture",
                                            ].map((hint) => (
                                                <button
                                                    key={hint}
                                                    onClick={() => setInput(hint)}
                                                    className="text-xs px-3 py-2 rounded-xl transition-all"
                                                    style={{
                                                        background: "var(--bg-elevated)",
                                                        color: "var(--text-muted)",
                                                        border: "1px solid var(--border)",
                                                    }}
                                                >
                                                    {hint}
                                                </button>
                                            ))}
                                        </div>
                                    </>
                                )}
                            </div>
                        )}

                        {messages.map((msg, i) => (
                            <div
                                key={i}
                                className={`flex gap-3 msg-enter ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                            >
                                {msg.role === "assistant" && (
                                    <div
                                        className="w-7 h-7 rounded-xl flex-shrink-0 flex items-center justify-center mt-1"
                                        style={{ background: "var(--accent)" }}
                                    >
                                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5">
                                            <path d="M12 2L2 7l10 5 10-5-10-5z" />
                                            <path d="M2 17l10 5 10-5" />
                                            <path d="M2 12l10 5 10-5" />
                                        </svg>
                                    </div>
                                )}

                                <div className={`max-w-[75%] ${msg.role === "user" ? "items-end" : "items-start"} flex flex-col`}>
                                    {msg.role === "assistant" && (
                                        <div className="flex items-center gap-2 mb-1.5">
                                            <span className="text-[10px] font-bold" style={{ color: "var(--text-muted)" }}>
                                                Agent
                                            </span>
                                            {!msg.streaming && <IntentBadge intent={msg.intent} />}
                                        </div>
                                    )}

                                    <div
                                        className="rounded-2xl px-4 py-3"
                                        style={{
                                            background:
                                                msg.role === "user"
                                                    ? "var(--accent)"
                                                    : "var(--bg-elevated)",
                                            border: msg.role === "user" ? "none" : "1px solid var(--border)",
                                            borderBottomRightRadius: msg.role === "user" ? "6px" : undefined,
                                            borderBottomLeftRadius: msg.role === "assistant" ? "6px" : undefined,
                                        }}
                                    >
                                        <MessageBody
                                            content={msg.content || (msg.streaming ? " " : "…")}
                                            intent={msg.intent}
                                            streaming={msg.streaming}
                                        />
                                    </div>

                                    {/* TTS + Citations for assistant messages */}
                                    {!msg.streaming && msg.role === "assistant" && (
                                        <div className="flex items-center gap-2 mt-1.5">
                                            <button
                                                onClick={() => handleSpeak(msg.content, i)}
                                                className="flex items-center gap-1 px-2 py-1 rounded-lg text-[10px] font-semibold transition-all hover:bg-[var(--bg-elevated)]"
                                                style={{
                                                    color: speakingMsgIdx === i ? "var(--accent)" : "var(--text-dim)",
                                                    border: `1px solid ${speakingMsgIdx === i ? "rgba(124,106,247,0.4)" : "transparent"}`,
                                                }}
                                                title={speakingMsgIdx === i ? "Stop speaking" : "Read aloud"}
                                            >
                                                {speakingMsgIdx === i ? (
                                                    <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor" stroke="none">
                                                        <rect x="6" y="4" width="4" height="16" rx="1" />
                                                        <rect x="14" y="4" width="4" height="16" rx="1" />
                                                    </svg>
                                                ) : (
                                                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                                        <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
                                                        <path d="M15.54 8.46a5 5 0 010 7.07" />
                                                        <path d="M19.07 4.93a10 10 0 010 14.14" />
                                                    </svg>
                                                )}
                                                {speakingMsgIdx === i ? "Stop" : "Listen"}
                                            </button>
                                            <CitationList sources={msg.sources} />
                                        </div>
                                    )}
                                </div>

                                {msg.role === "user" && (
                                    <div
                                        className="w-7 h-7 rounded-xl flex-shrink-0 flex items-center justify-center mt-1 text-xs font-bold"
                                        style={{ background: "var(--bg-elevated)", color: "var(--text-muted)" }}
                                    >
                                        U
                                    </div>
                                )}
                            </div>
                        ))}
                        <div ref={bottomRef} />
                    </div>

                    {/* Input Area */}
                    <div className="px-6 pb-6">
                        {/* Page range filter */}
                        <div className="flex items-center gap-3 mb-3">
                            <span className="text-[10px] font-bold uppercase tracking-widest" style={{ color: "var(--text-dim)" }}>
                                Page range
                            </span>
                            <input
                                type="number"
                                value={pageStart}
                                onChange={(e) => setPageStart(e.target.value)}
                                placeholder="From"
                                className="w-16 text-xs px-2 py-1 rounded-lg outline-none"
                                style={{
                                    background: "var(--bg-elevated)",
                                    border: "1px solid var(--border)",
                                    color: "var(--text-primary)",
                                }}
                            />
                            <span style={{ color: "var(--text-dim)" }}>–</span>
                            <input
                                type="number"
                                value={pageEnd}
                                onChange={(e) => setPageEnd(e.target.value)}
                                placeholder="To"
                                className="w-16 text-xs px-2 py-1 rounded-lg outline-none"
                                style={{
                                    background: "var(--bg-elevated)",
                                    border: "1px solid var(--border)",
                                    color: "var(--text-primary)",
                                }}
                            />
                            {(pageStart || pageEnd) && (
                                <button
                                    onClick={() => { setPageStart(""); setPageEnd(""); }}
                                    className="text-[10px] font-bold"
                                    style={{ color: "var(--text-dim)" }}
                                >
                                    Clear
                                </button>
                            )}
                        </div>

                        <div
                            className="flex items-end gap-3 p-3 rounded-2xl transition-all"
                            style={{
                                background: "var(--bg-secondary)",
                                border: "1px solid var(--border-hover)",
                            }}
                        >
                            <input
                                type="file"
                                multiple
                                ref={fileInputRef}
                                className="hidden"
                                onChange={handleFileUpload}
                                accept=".pdf,.docx,.txt"
                            />
                            <button
                                onClick={() => fileInputRef.current?.click()}
                                disabled={loading}
                                className="w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0 transition-all hover:bg-[var(--bg-elevated)]"
                                style={{
                                    color: "var(--text-dim)",
                                    border: "1px solid var(--border)",
                                }}
                                title="Add files"
                            >
                                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                    <path d="M12 5v14M5 12h14" />
                                </svg>
                            </button>

                            <textarea
                                ref={textareaRef}
                                value={input}
                                onChange={(e) => setInput(e.target.value)}
                                onKeyDown={handleKeyDown}
                                placeholder="Ask a question, compare docs, or request a diagram…"
                                disabled={loading}
                                rows={1}
                                className="flex-1 resize-none bg-transparent outline-none text-sm leading-relaxed"
                                style={{ color: "var(--text-primary)", maxHeight: "160px" }}
                            />

                            {/* Mic button */}
                            <button
                                onClick={toggleListening}
                                disabled={loading}
                                className="w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0 transition-all"
                                style={{
                                    background: isListening ? "rgba(239,68,68,0.15)" : "transparent",
                                    color: isListening ? "#ef4444" : "var(--text-dim)",
                                    border: `1px solid ${isListening ? "rgba(239,68,68,0.4)" : "var(--border)"}`,
                                    animation: isListening ? "pulse 1.5s ease-in-out infinite" : "none",
                                }}
                                title={isListening ? "Stop listening" : "Voice input"}
                            >
                                {isListening ? (
                                    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" stroke="none">
                                        <rect x="6" y="4" width="4" height="16" rx="1" />
                                        <rect x="14" y="4" width="4" height="16" rx="1" />
                                    </svg>
                                ) : (
                                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                        <path d="M12 1a3 3 0 00-3 3v8a3 3 0 006 0V4a3 3 0 00-3-3z" />
                                        <path d="M19 10v2a7 7 0 01-14 0v-2" />
                                        <line x1="12" y1="19" x2="12" y2="23" />
                                        <line x1="8" y1="23" x2="16" y2="23" />
                                    </svg>
                                )}
                            </button>

                            {/* Send button */}
                            <button
                                onClick={handleSend}
                                disabled={!input.trim() || loading}
                                className="w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0 transition-all"
                                style={{
                                    background: input.trim() && !loading ? "var(--accent)" : "var(--bg-elevated)",
                                    color: input.trim() && !loading ? "white" : "var(--text-dim)",
                                }}
                            >
                                {loading ? (
                                    <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                        <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
                                    </svg>
                                ) : (
                                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                                        <path d="M22 2L11 13M22 2L15 22 11 13 2 9 22 2z" />
                                    </svg>
                                )}
                            </button>
                        </div>
                        <p className="text-center text-[9px] mt-2 font-bold uppercase tracking-[0.3em]"
                            style={{ color: "var(--text-dim)" }}>
                            Groq · RAG · pgvector
                        </p>
                    </div>
                </div>

                {/* Document Panel */}
                {showDocPanel && (
                    <aside
                        className="w-64 flex-shrink-0 border-l flex flex-col"
                        style={{ borderColor: "var(--border)", background: "var(--bg-secondary)" }}
                    >
                        <div className="p-4 border-b" style={{ borderColor: "var(--border)" }}>
                            <p className="text-[10px] font-bold uppercase tracking-widest" style={{ color: "var(--text-dim)" }}>
                                Documents
                            </p>
                            <button
                                onClick={() =>
                                    setSelectedDocs(
                                        allSelected ? [] : sessionDocs.map((d) => d.id)
                                    )
                                }
                                className="text-xs mt-1 font-semibold"
                                style={{ color: "var(--accent)" }}
                            >
                                {allSelected ? "Deselect all" : "Select all"}
                            </button>
                        </div>
                        <div className="flex-1 overflow-y-auto p-3 space-y-2">
                            {sessionDocs.map((doc: Document) => {
                                const selected = selectedDocs.includes(doc.id);
                                return (
                                    <div
                                        key={doc.id}
                                        onClick={() => toggleDoc(doc.id)}
                                        className="flex items-start gap-3 p-3 rounded-xl cursor-pointer transition-all"
                                        style={{
                                            background: selected ? "var(--accent-dim)" : "var(--bg-elevated)",
                                            border: `1px solid ${selected ? "rgba(124,106,247,0.4)" : "var(--border)"}`,
                                        }}
                                    >
                                        <div
                                            className="w-5 h-5 rounded-md flex-shrink-0 flex items-center justify-center mt-0.5"
                                            style={{
                                                background: selected ? "var(--accent)" : "var(--bg-surface)",
                                                border: `1px solid ${selected ? "var(--accent)" : "var(--border)"}`,
                                            }}
                                        >
                                            {selected && (
                                                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="3">
                                                    <path d="M20 6L9 17 4 12" />
                                                </svg>
                                            )}
                                        </div>
                                        <div className="min-w-0">
                                            <p className="text-xs font-semibold truncate" style={{ color: "var(--text-primary)" }}>
                                                {doc.title}
                                            </p>
                                            <p className="text-[10px]" style={{ color: "var(--text-muted)" }}>
                                                {doc.file_type.toUpperCase()} · {doc.total_pages} pages
                                            </p>
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    </aside>
                )}
            </div>

            {/* Voice Assistant Overlay */}
            {showVoice && (
                <VoiceAssistant
                    sessionId={session.id}
                    documentIds={selectedDocs}
                    onClose={() => setShowVoice(false)}
                />
            )}
        </div>
    );
}
