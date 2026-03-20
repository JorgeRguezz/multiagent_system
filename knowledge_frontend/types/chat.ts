export type ChatRole = "user" | "assistant"

export type ChatMessage = {
  id: string
  role: ChatRole
  content: string
}

export type EvidenceItem = {
  video_name?: string
  time_span?: string
  chunk_id?: string
  source?: string
  text?: string
  final_score?: number
}

export type DebugInfo = {
  confidence?: number
  confidence_band?: "low" | "medium" | "high"
  evidence?: EvidenceItem[]
  generation?: {
    thoughts?: string
    answer_raw?: string
    has_final_marker?: boolean
    raw_text?: string
    verifier_enabled?: boolean
    final_answer_source?: "verified" | "raw_generation"
  }
  retrieval_counts?: Record<string, number>
  intent?: {
    normalized_query?: string
    is_cross_video?: boolean
    is_temporal?: boolean
    is_visual_detail?: boolean
    entity_focus_terms?: string[]
  }
  verification?: Record<string, unknown>
  final_evidence_count?: number
  [key: string]: unknown
}

export type ChatResponse = {
  answer: string
  debug?: DebugInfo
}
