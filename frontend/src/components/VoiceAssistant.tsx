"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { getDeepgramToken, speakText, streamChat } from "@/lib/api";
import { Source } from "@/lib/types";

interface Props {
    sessionId: number;
    documentIds: number[];
    onClose: () => void;
}

type VoiceState = "idle" | "listening" | "processing" | "speaking";

interface VoiceMessage {
    role: "user" | "assistant";
    content: string;
}

export default function VoiceAssistant({ sessionId, documentIds, onClose }: Props) {
    const [state, setState] = useState<VoiceState>("idle");
    const [transcript, setTranscript] = useState("");
    const [messages, setMessages] = useState<VoiceMessage[]>([]);
    const [error, setError] = useState<string | null>(null);
    const socketRef = useRef<WebSocket | null>(null);
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const streamRef = useRef<MediaStream | null>(null);
    const audioRef = useRef<HTMLAudioElement | null>(null);
    const silenceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
    const finalTranscriptRef = useRef("");
    const messagesEndRef = useRef<HTMLDivElement>(null);

    // Scroll to bottom
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages]);

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            stopListening();
            if (audioRef.current) {
                audioRef.current.pause();
                audioRef.current = null;
            }
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    const stopListening = useCallback(() => {
        if (silenceTimerRef.current) {
            clearTimeout(silenceTimerRef.current);
            silenceTimerRef.current = null;
        }
        if (socketRef.current) {
            socketRef.current.close();
            socketRef.current = null;
        }
        if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
            mediaRecorderRef.current.stop();
            mediaRecorderRef.current = null;
        }
        if (streamRef.current) {
            streamRef.current.getTracks().forEach(t => t.stop());
            streamRef.current = null;
        }
    }, []);

    const processQuery = useCallback(async (query: string) => {
        if (!query.trim()) {
            setState("idle");
            return;
        }

        // Add user message
        setMessages(prev => [...prev, { role: "user", content: query }]);
        setState("processing");
        setTranscript("");

        // Stream RAG response
        let fullResponse = "";
        await streamChat(
            sessionId,
            { query, document_ids: documentIds },
            (token) => {
                fullResponse += token;
                // Update live
                setMessages(prev => {
                    const updated = [...prev];
                    const lastIdx = updated.length - 1;
                    if (lastIdx >= 0 && updated[lastIdx].role === "assistant") {
                        updated[lastIdx] = { ...updated[lastIdx], content: fullResponse };
                    } else {
                        updated.push({ role: "assistant", content: fullResponse });
                    }
                    return updated;
                });
            },
            (_sources: Source[], _intent: string) => {
                // Done — speak the response
                if (fullResponse) {
                    speakResponse(fullResponse);
                } else {
                    setState("idle");
                }
            },
            (errMsg) => {
                setError(errMsg);
                setState("idle");
            }
        );
    }, [sessionId, documentIds]);

    const speakResponse = useCallback(async (text: string) => {
        setState("speaking");
        try {
            const audioBuffer = await speakText(text);
            const blob = new Blob([audioBuffer], { type: "audio/mpeg" });
            const url = URL.createObjectURL(blob);
            const audio = new Audio(url);
            audioRef.current = audio;

            audio.onended = () => {
                URL.revokeObjectURL(url);
                audioRef.current = null;
                setState("idle");
            };
            audio.onerror = () => {
                URL.revokeObjectURL(url);
                audioRef.current = null;
                setState("idle");
            };

            await audio.play();
        } catch (err) {
            console.error("TTS error:", err);
            setState("idle");
        }
    }, []);

    const startListening = useCallback(async () => {
        setError(null);
        finalTranscriptRef.current = "";
        setTranscript("");

        try {
            // Get Deepgram token
            const apiKey = await getDeepgramToken();

            // Get mic access
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            streamRef.current = stream;

            // Connect to Deepgram WebSocket
            const socket = new WebSocket(
                "wss://api.deepgram.com/v1/listen?model=nova-2&smart_format=true&interim_results=true&utterance_end_ms=1500&vad_events=true",
                ["token", apiKey]
            );
            socketRef.current = socket;

            socket.onopen = () => {
                setState("listening");

                // Set up MediaRecorder
                const mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
                mediaRecorderRef.current = mediaRecorder;

                mediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 0 && socket.readyState === WebSocket.OPEN) {
                        socket.send(event.data);
                    }
                };

                mediaRecorder.start(250); // Send data every 250ms
            };

            socket.onmessage = (event) => {
                const data = JSON.parse(event.data);

                // Handle Utterance End event — user stopped talking
                if (data.type === "UtteranceEnd") {
                    if (finalTranscriptRef.current.trim()) {
                        const query = finalTranscriptRef.current.trim();
                        stopListening();
                        processQuery(query);
                    }
                    return;
                }

                // Handle transcript results
                if (data.channel?.alternatives?.[0]) {
                    const alt = data.channel.alternatives[0];
                    const transcriptText = alt.transcript;

                    if (transcriptText) {
                        if (data.is_final) {
                            finalTranscriptRef.current += " " + transcriptText;
                            setTranscript(finalTranscriptRef.current.trim());

                            // Reset silence timer
                            if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current);
                            silenceTimerRef.current = setTimeout(() => {
                                if (finalTranscriptRef.current.trim()) {
                                    const query = finalTranscriptRef.current.trim();
                                    stopListening();
                                    processQuery(query);
                                }
                            }, 2000);
                        } else {
                            // Interim result
                            setTranscript(
                                (finalTranscriptRef.current + " " + transcriptText).trim()
                            );
                        }
                    }
                }
            };

            socket.onerror = (e) => {
                console.error("Deepgram WebSocket error:", e);
                setError("Voice connection error. Check your Deepgram API key.");
                stopListening();
                setState("idle");
            };

            socket.onclose = () => {
                // Normal close
            };

        } catch (err: any) {
            console.error("Start listening error:", err);
            setError(err.message || "Failed to start voice input");
            setState("idle");
        }
    }, [stopListening, processQuery]);

    const handleMicClick = useCallback(() => {
        if (state === "listening") {
            // Stop and process what we have
            const query = finalTranscriptRef.current.trim();
            stopListening();
            if (query) {
                processQuery(query);
            } else {
                setState("idle");
            }
        } else if (state === "speaking") {
            // Stop TTS
            if (audioRef.current) {
                audioRef.current.pause();
                audioRef.current = null;
            }
            setState("idle");
        } else if (state === "idle") {
            startListening();
        }
    }, [state, stopListening, processQuery, startListening]);

    const stateConfig = {
        idle: { label: "Tap to speak", color: "var(--accent)", pulse: false },
        listening: { label: "Listening…", color: "#ef4444", pulse: true },
        processing: { label: "Thinking…", color: "#f59e0b", pulse: true },
        speaking: { label: "Speaking…", color: "#10b981", pulse: true },
    };

    const { label, color, pulse } = stateConfig[state];

    return (
        <div
            className="fixed inset-0 z-50 flex flex-col"
            style={{ background: "rgba(0,0,0,0.92)", backdropFilter: "blur(20px)" }}
        >
            {/* Header */}
            <div className="flex items-center justify-between px-6 py-4">
                <div className="flex items-center gap-3">
                    <div
                        className="w-8 h-8 rounded-xl flex items-center justify-center"
                        style={{ background: "var(--accent)" }}
                    >
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5">
                            <path d="M12 1a3 3 0 00-3 3v8a3 3 0 006 0V4a3 3 0 00-3-3z" />
                            <path d="M19 10v2a7 7 0 01-14 0v-2" />
                        </svg>
                    </div>
                    <span className="text-sm font-bold" style={{ color: "var(--text-primary)" }}>
                        Voice Assistant
                    </span>
                </div>
                <button
                    onClick={onClose}
                    className="w-9 h-9 rounded-xl flex items-center justify-center transition-all hover:bg-white/10"
                    style={{ color: "var(--text-muted)" }}
                >
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M18 6L6 18M6 6l12 12" />
                    </svg>
                </button>
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
                {messages.length === 0 && state === "idle" && (
                    <div className="flex flex-col items-center justify-center h-full text-center">
                        <div
                            className="w-20 h-20 rounded-3xl flex items-center justify-center mb-6"
                            style={{
                                background: "rgba(124,106,247,0.15)",
                                border: "1px solid rgba(124,106,247,0.3)",
                            }}
                        >
                            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"
                                style={{ color: "var(--accent)" }}>
                                <path d="M12 1a3 3 0 00-3 3v8a3 3 0 006 0V4a3 3 0 00-3-3z" />
                                <path d="M19 10v2a7 7 0 01-14 0v-2" />
                                <line x1="12" y1="19" x2="12" y2="23" />
                                <line x1="8" y1="23" x2="16" y2="23" />
                            </svg>
                        </div>
                        <h2 className="text-lg font-bold mb-2" style={{ color: "var(--text-primary)" }}>
                            Talk to your documents
                        </h2>
                        <p className="text-xs max-w-xs" style={{ color: "var(--text-muted)" }}>
                            Tap the microphone and ask a question about your uploaded documents.
                            The assistant will respond with voice.
                        </p>
                    </div>
                )}

                {messages.map((msg, i) => (
                    <div
                        key={i}
                        className={`flex gap-3 ${msg.role === "user" ? "justify-end" : "justify-start"}`}
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
                        <div
                            className="max-w-[75%] rounded-2xl px-4 py-3"
                            style={{
                                background: msg.role === "user" ? "var(--accent)" : "rgba(255,255,255,0.06)",
                                border: msg.role === "user" ? "none" : "1px solid rgba(255,255,255,0.08)",
                                borderBottomRightRadius: msg.role === "user" ? "6px" : undefined,
                                borderBottomLeftRadius: msg.role === "assistant" ? "6px" : undefined,
                            }}
                        >
                            <p className="text-sm leading-relaxed whitespace-pre-wrap" style={{ color: "white" }}>
                                {msg.content}
                            </p>
                        </div>
                    </div>
                ))}
                <div ref={messagesEndRef} />
            </div>

            {/* Live transcript */}
            {transcript && state === "listening" && (
                <div className="px-6 pb-2">
                    <div
                        className="rounded-xl px-4 py-2 text-xs"
                        style={{
                            background: "rgba(255,255,255,0.04)",
                            border: "1px solid rgba(239,68,68,0.3)",
                            color: "var(--text-muted)",
                        }}
                    >
                        <span className="text-[10px] font-bold uppercase tracking-widest mr-2" style={{ color: "#ef4444" }}>
                            Live
                        </span>
                        {transcript}
                    </div>
                </div>
            )}

            {/* Error */}
            {error && (
                <div className="px-6 pb-2">
                    <p className="text-xs text-red-400 text-center">{error}</p>
                </div>
            )}

            {/* Mic Button */}
            <div className="flex flex-col items-center gap-3 pb-8 pt-4">
                <button
                    onClick={handleMicClick}
                    disabled={state === "processing"}
                    className="relative w-20 h-20 rounded-full flex items-center justify-center transition-all"
                    style={{
                        background: color,
                        opacity: state === "processing" ? 0.6 : 1,
                        boxShadow: pulse ? `0 0 0 0 ${color}` : "none",
                        animation: pulse ? "voice-pulse 2s ease-in-out infinite" : "none",
                    }}
                >
                    {state === "listening" ? (
                        /* Stop / pause icon */
                        <svg width="28" height="28" viewBox="0 0 24 24" fill="white" stroke="none">
                            <rect x="6" y="4" width="4" height="16" rx="1" />
                            <rect x="14" y="4" width="4" height="16" rx="1" />
                        </svg>
                    ) : state === "processing" ? (
                        /* Spinner */
                        <svg className="animate-spin" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2">
                            <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
                        </svg>
                    ) : state === "speaking" ? (
                        /* Speaker / stop icon */
                        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2">
                            <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" fill="white" />
                            <path d="M15.54 8.46a5 5 0 010 7.07" />
                            <path d="M19.07 4.93a10 10 0 010 14.14" />
                        </svg>
                    ) : (
                        /* Mic icon */
                        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2">
                            <path d="M12 1a3 3 0 00-3 3v8a3 3 0 006 0V4a3 3 0 00-3-3z" />
                            <path d="M19 10v2a7 7 0 01-14 0v-2" />
                            <line x1="12" y1="19" x2="12" y2="23" />
                            <line x1="8" y1="23" x2="16" y2="23" />
                        </svg>
                    )}
                </button>
                <span
                    className="text-xs font-bold uppercase tracking-widest"
                    style={{ color: "var(--text-muted)" }}
                >
                    {label}
                </span>
            </div>
        </div>
    );
}
