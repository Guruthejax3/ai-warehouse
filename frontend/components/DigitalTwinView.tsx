"use client";

// Digital twin: lightweight isometric canvas rendering of the warehouse layout
// with zone heat overlays, active hazard markers, and object position indicators.

import { useEffect, useRef, useState } from "react";
import { getDigitalTwin, DigitalTwinState, RISK_STYLES } from "@/lib/api";

const BAYS = ["bay_0", "bay_1", "bay_2", "bay_3", "bay_4", "bay_5", "bay_6", "bay_7"];

// Isometric bay positions (2x4 grid, isometric projection)
const BAY_POSITIONS: Record<string, [number, number]> = {
  bay_0: [0.15, 0.25],
  bay_1: [0.40, 0.25],
  bay_2: [0.65, 0.25],
  bay_3: [0.88, 0.25],
  bay_4: [0.15, 0.60],
  bay_5: [0.40, 0.60],
  bay_6: [0.65, 0.60],
  bay_7: [0.88, 0.60],
};

const STATUS_COLORS: Record<string, string> = {
  normal: "#10b981",
  warning: "#f59e0b",
  critical: "#ef4444",
};

export default function DigitalTwinView() {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [data, setData] = useState<DigitalTwinState | null>(null);
  const [hoveredBay, setHoveredBay] = useState<string | null>(null);

  useEffect(() => {
    getDigitalTwin(1).then(setData).catch(() => {});
  }, []);

  // Draw the isometric warehouse
  useEffect(() => {
    if (!data) return;
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;

    const dpr = window.devicePixelRatio || 1;
    const w = wrap.clientWidth;
    const h = 280;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    // Background grid (warehouse floor)
    ctx.strokeStyle = "#e2e8f0";
    ctx.lineWidth = 0.5;
    for (let x = 0; x < w; x += 40) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }
    for (let y = 0; y < h; y += 40) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    // Aisle label
    ctx.fillStyle = "#94a3b8";
    ctx.font = "11px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("AISLE", w / 2, h / 2);

    // Draw each bay
    for (const bay of BAYS) {
      const [bx, by] = BAY_POSITIONS[bay];
      const x = bx * w;
      const y = by * h;
      const bw = w * 0.18;
      const bh = h * 0.28;

      const zone = data.zones[bay];
      const status = zone?.status ?? "normal";
      const color = STATUS_COLORS[status];

      // Bay rectangle
      ctx.fillStyle = color + "22";
      ctx.strokeStyle = color;
      ctx.lineWidth = hoveredBay === bay ? 3 : 1.5;
      ctx.beginPath();
      ctx.roundRect(x - bw / 2, y - bh / 2, bw, bh, 4);
      ctx.fill();
      ctx.stroke();

      // Bay label
      ctx.fillStyle = "#475569";
      ctx.font = "bold 11px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(bay.replace("_", " "), x, y - bh / 2 + 14);

      // Risk score
      if (zone && zone.events > 0) {
        ctx.fillStyle = color;
        ctx.font = "bold 13px sans-serif";
        ctx.fillText(zone.max_risk.toFixed(2), x, y + 4);

        // Event count
        ctx.fillStyle = "#94a3b8";
        ctx.font = "9px sans-serif";
        ctx.fillText(`${zone.events} events`, x, y + 18);
      } else {
        ctx.fillStyle = "#cbd5e1";
        ctx.font = "10px sans-serif";
        ctx.fillText("clear", x, y + 4);
      }

      // Pulse effect for critical bays
      if (status === "critical") {
        const t = (Date.now() % 2000) / 2000;
        const pulse = Math.sin(t * Math.PI * 2) * 0.3 + 0.7;
        ctx.strokeStyle = `rgba(239, 68, 68, ${pulse * 0.5})`;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.roundRect(x - bw / 2 - 4, y - bh / 2 - 4, bw + 8, bh + 8, 6);
        ctx.stroke();
      }
    }

    // Hazard markers (pulsing red dots for active hazards)
    for (const hazard of data.active_hazards) {
      const pos = BAY_POSITIONS[hazard.zone_id];
      if (!pos) continue;
      const hx = pos[0] * w + w * 0.06;
      const hy = pos[1] * h - h * 0.1;
      const t = (Date.now() % 1500) / 1500;
      const pulse = Math.sin(t * Math.PI * 2) * 2 + 6;

      ctx.fillStyle = "#ef4444";
      ctx.beginPath();
      ctx.arc(hx, hy, pulse, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "white";
      ctx.beginPath();
      ctx.arc(hx, hy, 2, 0, Math.PI * 2);
      ctx.fill();
    }
  }, [data, hoveredBay]);

  // Re-render for pulse animation
  useEffect(() => {
    if (!data) return;
    const hasCritical = Object.values(data.zones).some(
      (z) => z.status === "critical"
    );
    if (!hasCritical && data.active_hazards.length === 0) return;

    let raf: number;
    const animate = () => {
      // trigger redraw by setting state (no-op, just forces effect)
      setData((prev) => prev);
      raf = requestAnimationFrame(animate);
    };
    raf = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(raf);
  }, [data]);

  // Mouse hover detection
  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || !data) return;
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left) / rect.width;
    const my = (e.clientY - rect.top) / rect.height;

    let found: string | null = null;
    for (const bay of BAYS) {
      const [bx, by] = BAY_POSITIONS[bay];
      if (Math.abs(mx - bx) < 0.09 && Math.abs(my - by) < 0.14) {
        found = bay;
        break;
      }
    }
    setHoveredBay(found);
  };

  return (
    <section className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <header className="mb-3 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-slate-700">
          Digital twin — warehouse overview
        </h2>
        {data && (
          <div className="flex items-center gap-3 text-[11px] text-slate-400">
            <span>{data.total_events} events</span>
            <span
              className={
                data.overall_risk > 0.7
                  ? "text-red-600"
                  : data.overall_risk > 0.4
                    ? "text-amber-600"
                    : "text-emerald-600"
              }
            >
              risk: {data.overall_risk.toFixed(2)}
            </span>
          </div>
        )}
      </header>

      <div ref={wrapRef} className="w-full">
        <canvas
          ref={canvasRef}
          style={{ width: "100%", height: 280, display: "block", cursor: "crosshair" }}
          onMouseMove={handleMouseMove}
          onMouseLeave={() => setHoveredBay(null)}
        />
      </div>

      {/* Hovered bay tooltip */}
      {hoveredBay && data?.zones[hoveredBay] && (
        <div className="mt-2 rounded-lg bg-slate-50 px-3 py-2 text-xs text-slate-600">
          <span className="font-semibold">{hoveredBay.replace("_", " ")}</span>:{" "}
          {data.zones[hoveredBay].events} events, max risk{" "}
          {data.zones[hoveredBay].max_risk.toFixed(2)} ({data.zones[hoveredBay].status})
        </div>
      )}

      {/* Legend */}
      <div className="mt-2 flex items-center gap-3 text-[11px] text-slate-400">
        <span className="flex items-center gap-1">
          <span className="h-2 w-2 rounded-sm bg-emerald-500" /> normal
        </span>
        <span className="flex items-center gap-1">
          <span className="h-2 w-2 rounded-sm bg-amber-500" /> warning
        </span>
        <span className="flex items-center gap-1">
          <span className="h-2 w-2 rounded-sm bg-red-500" /> critical
        </span>
        <span className="flex items-center gap-1">
          <span className="h-2 w-2 rounded-full bg-red-500 animate-pulse" /> active hazard
        </span>
      </div>
    </section>
  );
}
