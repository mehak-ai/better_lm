// Shared API types

export interface Document {
    id: number;
    title: string;
    file_type: string;
    uploaded_at: string;
    total_pages: number;
}

export interface ChatSession {
    id: number;
    title: string;
    created_at: string;
    documents: Document[];
}

export interface Source {
    chunk_id: number;
    document_id: number;
    document_title: string;
    page_number: number;
    score: number;
    excerpt: string;
}

export interface Message {
    id?: number;
    role: "user" | "assistant";
    content: string;
    sources: Source[];
    intent: "rag" | "compare" | "mermaid";
    created_at?: string;
    streaming?: boolean;
}

export interface ChatRequest {
    query: string;
    document_ids?: number[];
    page_start?: number | null;
    page_end?: number | null;
}
