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