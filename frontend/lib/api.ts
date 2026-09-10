// ReplayTwin frontend -> backend API client.
// Point NEXT_PUBLIC_API_URL at the backend when it isn't on localhost:8000
// (e.g. docker compose: http://backend:8000).

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export const WS_URL =
  (process.env.NEXT_PUBLIC_WS_URL ?? API_BASE.replace(/^http/, "ws")) +
  "/ws/events";

// ---------------------------------------------------------------------------
// Types mirroring the backend contracts (backend/main.py, pipeline/types.py)
// ---------------------------------------------------------------------------

export interface RiskEvent {
  event_id: string;
  created_at?: string | null;
  timestamp_sec: number;
  frame_idx: number;
  source_video: string;
  behavior_class: string;
  risk_score: number;
  risk_level: "low" | "medium" | "high" | "critical";
  confidence?: number | null;
  dtw_distance?: number | null;
  physics_valid: boolean;
  physics_severity?: number | null;
  justification: string;
  evidence_clip_path?: string | null;
  zone_id: string;
}

export interface ReplayPayload {
  event_id: string;
  behavior_class: string;
  duration_sec: number;
  normalized: boolean;
  actual_path: [number, number][];
  correct_path: [number, number][];
  frames: {
    frame_idx: number;
    t: number;
    actual: [number, number] | null;
    correct: [number, number] | null;
  }[];
}

export const RISK_ORDER: Record<RiskEvent["risk_level"], number> = {
  low: 0,
  medium: 1,
  high: 2,
  critical: 3,
};

export const RISK_STYLES: Record<
  RiskEvent["risk_level"],
  { badge: string; dot: string; heat: string }
> = {
  low: {
    badge: "bg-emerald-100 text-emerald-800 border-emerald-200",
    dot: "bg-emerald-400",
    heat: "#34d399",
  },
  medium: {
    badge: "bg-amber-100 text-amber-800 border-amber-200",
    dot: "bg-amber-400",
    heat: "#fbbf24",
  },
  high: {
    badge: "bg-orange-100 text-orange-800 border-orange-200",
    dot: "bg-orange-500",
    heat: "#fb923c",
  },
  critical: {
    badge: "bg-red-100 text-red-800 border-red-200",
    dot: "bg-red-600",
    heat: "#ef4444",
  },
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    cache: "no-store",
  });
  if (!res.ok) {
    throw new Error(`API ${path} -> ${res.status} ${await res.text()}`);
  }
  return res.json() as Promise<T>;
}

export function getEvents(limit = 200): Promise<RiskEvent[]> {
  return request<RiskEvent[]>(`/api/events?limit=${limit}`);
}

export function getReplay(eventId: string): Promise<ReplayPayload> {
  return request<ReplayPayload>(`/api/replay/${encodeURIComponent(eventId)}`);
}

export function getEventDetail(eventId: string): Promise<{
  event: RiskEvent;
  replay: ReplayPayload;
}> {
  return request(`/api/events/${encodeURIComponent(eventId)}`);
}

export function getBehaviors(): Promise<{ behavior_classes: string[] }> {
  return request("/api/behaviors");
}

export async function askAssistant(message: string): Promise<string> {
  const data = await request<{ message: string }>("/api/assistant", {
    method: "POST",
    body: JSON.stringify({ message }),
  });
  return data.message;
}

// ---------------------------------------------------------------------------
// Analytics & innovation endpoints (Feature 5/6 + innovations)
// ---------------------------------------------------------------------------

export interface ShiftSummary {
  window_days: number;
  total_events: number;
  serious_count: number;
  max_risk: number;
  avg_risk: number;
  zones_affected: number;
  behaviors: { behavior_class: string; n: number; avg_risk: number }[];
  top_bays: { zone_id: string; n: number; max_risk: number }[];
  narrative: string;
}

export interface TrendDay {
  day: string;
  n: number;
  serious: number;
  avg_risk: number;
}

export interface RecurringBehavior {
  behavior_class: string;
  occurrences: number;
  distinct_bays: number;
  bays: string[];
  max_risk: number;
  recurring: boolean;
}

export interface TrainingUnit {
  behavior_class: string;
  training_unit: string;
  category: string;
  process_fix: string;
}

export interface CorrectiveAction {
  behavior_class: string;
  risk_level: string;
  immediate_action: string;
  process_fix: string;
  training_unit: string;
  category: string;
  auto_action: boolean;
  auto_note: string;
}

export interface DigitalTwinState {
  zones: Record<string, {
    zone_id: string;
    events: number;
    max_risk: number;
    avg_risk: number;
    status: "normal" | "warning" | "critical";
  }>;
  active_hazards: {
    event_id: string;
    zone_id: string;
    behavior_class: string;
    risk_level: string;
    risk_score: number;
  }[];
  total_events: number;
  overall_risk: number;
  window_days: number;
}

export function getShiftSummary(windowDays = 1): Promise<ShiftSummary> {
  return request<ShiftSummary>(`/api/analytics/shift-summary?window_days=${windowDays}`);
}

export function getTrend(windowDays = 7): Promise<{ days: TrendDay[]; window_days: number }> {
  return request(`/api/analytics/trend?window_days=${windowDays}`);
}

export function getRecurring(windowDays = 7): Promise<{ behaviors: RecurringBehavior[] }> {
  return request(`/api/analytics/recurring?window_days=${windowDays}`);
}

export function getTraining(): Promise<{ training_units: TrainingUnit[] }> {
  return request("/api/analytics/training");
}

export function getCorrectiveActions(): Promise<{ recommendations: CorrectiveAction[] }> {
  return request("/api/analytics/corrective-actions");
}

export function getDigitalTwin(windowDays = 1): Promise<DigitalTwinState> {
  return request(`/api/digital-twin?window_days=${windowDays}`);
}