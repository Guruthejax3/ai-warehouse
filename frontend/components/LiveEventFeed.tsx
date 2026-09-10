"use client";

// Live event feed: REST snapshot on mount + WebSocket updates as the
// backend ingests clips in real time.

import { useEffect, useRef, useState } from "react";
import { RISK_STYLES, WS_URL, getEvents, RiskEvent } from "@/lib/api";
import Link from "next/link";

export default function LiveEventFeed() {
  const [events, setEvents] = useState<RiskEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    getEvents().then(setEvents).catch(() => {});
  }, []);

  useEffect(() => {
    let retry = 0;
    let alive = true;

    function connect() {
      if (!alive) return;
      let ws: WebSocket;
      try {
        ws = new WebSocket(WS_URL);
      } catch {
        return;
      }
      wsRef.current = ws;
      ws.onopen = () => {
        retry = 0;
        setConnected(true);
      };
      ws.onmessage = (msg) => {
        try {
          const payload = JSON.parse(msg.data as string);
          if (payload?.type === "event" && payload.data) {
            setEvents((prev) =>
              [payload.data as RiskEvent, ...prev.filter((e) => e.event_id !== payload.data.event_id)].slice(0, 200),
            );
          }
        } catch {
          /* ignore malformed frames */
        }
      };
      ws.onclose = () => {
        setConnected(false);
        if (!alive) return;
        const delay = Math.min(3000 * 2 ** retry, 15000);
        retry += 1;
        setTimeout(connect, delay);
      };
    }

    connect();
    return () => {
      alive = false;
      wsRef.current?.close();
    };
  }, []);

  const shown = events.slice(0, 8);

  return (
    <section className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <header className="mb-3 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-slate-700">Live event feed</h2>
        <span
          className={`inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-xs font-medium ${
            connected ? "bg-emerald-100 text-emerald-700" : "bg-slate-100 text-slate-500"
          }`}
        >
          <span className={`h-2 w-2 rounded-full ${connected ? "animate-pulse bg-emerald-500" : "bg-slate-400"}`} />
          {connected ? "live" : "reconnecting…"}
        </span>
      </header>

      {shown.length === 0 ? (
        <p className="rounded-lg bg-slate-50 px-3 py-6 text-center text-sm text-slate-400">
          No events yet — ingest a clip on the backend to see detections here.
        </p>
      ) : (
        <ul className="space-y-2">
          {shown.map((ev) => {
            const s = RISK_STYLES[ev.risk_level];
            return (
              <li key={ev.event_id} className="rounded-lg border border-slate-100 bg-slate-50/60 px-3 py-2">
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <span className={`h-2 w-2 rounded-full ${s.dot}`} />
                    <Link
                      href={`/replay/${ev.event_id}`}
                      className="text-sm font-medium text-slate-800 hover:text-slate-950 hover:underline"
                    >
                      {ev.behavior_class.replace(/_/g, " ")}
                    </Link>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-slate-400">
                      {ev.timestamp_sec.toFixed(1)}s · {ev.zone_id}
                    </span>
                    <span className={`rounded-full border px-2 py-0.5 text-xs font-semibold ${s.badge}`}>
                      {ev.risk_level}
                    </span>
                  </div>
                </div>
                <p className="mt-1 text-xs text-slate-500">
                  {ev.justification.length > 140 ? ev.justification.slice(0, 140) + "…" : ev.justification}
                </p>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}