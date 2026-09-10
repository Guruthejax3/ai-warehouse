"use client";

// Replay viewer: 2D Canvas overlay of the actual (red) vs correct-technique
// (green) trajectory for one event. Fetches GET /api/events/{id}, which
// returns event metadata + the replay payload (paths normalized to 0-1).

import { useEffect, useRef, useState } from "react";
import { getEventDetail, ReplayPayload, RISK_STYLES, RiskEvent } from "@/lib/api";

const PAD = 24;

export default function ReplayViewer({ eventId }: { eventId: string }) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number>(0);
  const tickRef = useRef(0); // 0-100

  const [data, setData] = useState<{ event: RiskEvent; replay: ReplayPayload } | null>(null);
  const [playing, setPlaying] = useState(false);
  const [tick, setTick] = useState(0);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getEventDetail(eventId)
      .then(setData)
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load replay"));
  }, [eventId]);

  // Draw the current frame; re-draw when data, playing, or tick changes.
  useEffect(() => {
    if (!data) return;
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;

    const dpr = window.devicePixelRatio || 1;
    const w = wrap.clientWidth;
    const h = 320;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const W = Math.max(1, w - PAD * 2);
    const H = Math.max(1, h - PAD * 2);
    const px = (x: number) => PAD + x * W;
    const py = (y: number) => PAD + y * H;

    const drawPath = (path: [number, number][], color: string, headIdx: number) => {
      if (path.length === 0) return;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(px(path[0][0]), py(path[0][1]));
      for (let i = 1; i < path.length; i++) ctx.lineTo(px(path[i][0]), py(path[i][1]));
      ctx.stroke();
      // start marker
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(px(path[0][0]), py(path[0][1]), 4, 0, Math.PI * 2);
      ctx.fill();
      // head dot at the sweep position
      const i = Math.max(0, Math.min(headIdx, path.length - 1));
      ctx.beginPath();
      ctx.arc(px(path[i][0]), py(path[i][1]), 5, 0, Math.PI * 2);
      ctx.fill();
    };

    const actual = data.replay.actual_path;
    const sweep = Math.round((tick / 100) * (actual.length - 1));
    drawPath(actual, "#dc2626", sweep);
    drawPath(data.replay.correct_path, "#16a34a", 0);

    ctx.strokeStyle = "#94a3b8";
    ctx.lineWidth = 1;
    ctx.strokeRect(PAD, PAD, W, H);
  }, [data, tick]);

  useEffect(() => () => cancelAnimationFrame(rafRef.current), []);

  function play() {
    if (!data || data.replay.actual_path.length < 2 || playing) return;
    setPlaying(true);
    cancelAnimationFrame(rafRef.current);
    const started = performance.now();
    const dur = Math.max(data.replay.duration_sec, 0.5) * 1000;
    const step = () => {
      const t = Math.min(1, (performance.now() - started) / dur);
      tickRef.current = Math.round(t * 100);
      setTick(tickRef.current);
      if (t < 1) {
        rafRef.current = requestAnimationFrame(step);
      } else {
        setPlaying(false);
        tickRef.current = 0;
        setTick(0);
      }
    };
    rafRef.current = requestAnimationFrame(step);
  }

  if (error) {
    return (
      <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
        {error}
      </div>
    );
  }
  if (!data) {
    return (
      <div className="rounded-xl border border-slate-200 bg-white p-4 text-sm text-slate-400">
        Loading replay…
      </div>
    );
  }

  const { event, replay } = data;
  const s = RISK_STYLES[event.risk_level];

  return (
    <section className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <header className="mb-3 flex flex-wrap items-start justify-between gap-2">
        <div>
          <h2 className="text-sm font-semibold text-slate-700">
            Replay — {replay.behavior_class.replace(/_/g, " ")}
          </h2>
          <div className="mt-1 flex items-center gap-2">
            <span className={`rounded-full border px-2 py-0.5 text-xs font-semibold ${s.badge}`}>
              {event.risk_level} · {event.risk_score.toFixed(2)}
            </span>
            <span className="text-xs text-slate-400">
              {event.timestamp_sec.toFixed(1)}s · {event.zone_id}
            </span>
          </div>
          {event.justification && (
            <p className="mt-1 max-w-xl text-xs text-slate-500">{event.justification}</p>
          )}
        </div>
        <button
          onClick={play}
          disabled={replay.actual_path.length < 2 || playing}
          className="rounded-lg bg-indigo-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-indigo-700 disabled:opacity-50"
        >
          {playing ? "Playing…" : "Play actual"}
        </button>
      </header>

      <div ref={wrapRef} className="w-full">
        <canvas ref={canvasRef} style={{ width: "100%", height: 320, display: "block" }} />
      </div>

      {/* Evidence clip player (if available) */}
      {event.evidence_clip_path && (
        <div className="mt-3 rounded-lg border border-slate-200 bg-slate-50 p-2">
          <p className="mb-1 text-[11px] font-medium text-slate-500">
            Evidence clip (face-blurred, trimmed)
          </p>
          <video
            controls
            className="w-full rounded-md"
            style={{ maxHeight: 200 }}
            src={`http://localhost:8000/api/evidence/${event.event_id}`}
          >
            Your browser does not support video playback.
          </video>
        </div>
      )}

      {/* Activity + sequence metadata */}
      {event.activity_type && event.activity_type !== "unknown" && (
        <div className="mt-2 flex items-center gap-2 text-[11px] text-slate-500">
          <span className="rounded bg-indigo-50 px-1.5 py-0.5 font-medium text-indigo-600">
            activity: {event.activity_type}
          </span>
        </div>
      )}

      <div className="mt-3 flex items-center gap-4 text-xs text-slate-500">
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full bg-red-600" /> actual
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full bg-emerald-600" /> correct technique
        </span>
        {replay.actual_path.length > 1 && (
          <input
            type="range"
            min={0}
            max={100}
            value={tick}
            onChange={(e) => {
              tickRef.current = Number(e.target.value);
              setTick(tickRef.current);
            }}
            className="ml-auto w-48"
          />
        )}
      </div>
    </section>
  );
}