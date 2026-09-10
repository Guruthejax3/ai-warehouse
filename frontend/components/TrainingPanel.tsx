"use client";

// Training recommendations: per-behavior training units organized by category.
// Ties directly to the corrective actions knowledge base.

import { useEffect, useState } from "react";
import { getTraining, TrainingUnit } from "@/lib/api";

const CATEGORY_COLORS: Record<string, string> = {
  handling: "bg-blue-100 text-blue-700",
  stacking: "bg-purple-100 text-purple-700",
  layout: "bg-cyan-100 text-cyan-700",
  safety: "bg-red-100 text-red-700",
  loading: "bg-amber-100 text-amber-700",
};

export default function TrainingPanel() {
  const [units, setUnits] = useState<TrainingUnit[]>([]);
  const [filter, setFilter] = useState<string | null>(null);

  useEffect(() => {
    getTraining()
      .then((r) => setUnits(r.training_units))
      .catch(() => {});
  }, []);

  const categories = [...new Set(units.map((u) => u.category))];
  const filtered = filter ? units.filter((u) => u.category === filter) : units;

  return (
    <section className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <header className="mb-3 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-slate-700">
          Training recommendations
        </h2>
        <span className="text-[11px] text-slate-400">{units.length} modules</span>
      </header>

      {/* Category filter pills */}
      <div className="mb-3 flex flex-wrap gap-1.5">
        <button
          onClick={() => setFilter(null)}
          className={`rounded-full px-2.5 py-0.5 text-[11px] font-medium transition-colors ${
            filter === null
              ? "bg-slate-800 text-white"
              : "bg-slate-100 text-slate-500 hover:bg-slate-200"
          }`}
        >
          all
        </button>
        {categories.map((cat) => (
          <button
            key={cat}
            onClick={() => setFilter(filter === cat ? null : cat)}
            className={`rounded-full px-2.5 py-0.5 text-[11px] font-medium transition-colors ${
              filter === cat
                ? CATEGORY_COLORS[cat] ?? "bg-slate-800 text-white"
                : "bg-slate-100 text-slate-500 hover:bg-slate-200"
            }`}
          >
            {cat}
          </button>
        ))}
      </div>

      {filtered.length === 0 ? (
        <p className="rounded-lg bg-slate-50 px-3 py-6 text-center text-sm text-slate-400">
          No training recommendations available.
        </p>
      ) : (
        <div className="space-y-2">
          {filtered.map((u) => (
            <div
              key={u.behavior_class}
              className="rounded-lg border border-slate-100 bg-slate-50/60 px-3 py-2"
            >
              <div className="flex items-center justify-between gap-2">
                <span className="text-xs font-medium text-slate-700">
                  {u.training_unit}
                </span>
                <span
                  className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${
                    CATEGORY_COLORS[u.category] ?? "bg-slate-100 text-slate-600"
                  }`}
                >
                  {u.category}
                </span>
              </div>
              <p className="mt-1 text-[11px] text-slate-500">
                Triggers: {u.behavior_class.replace(/_/g, " ")}
              </p>
              <p className="mt-0.5 text-[11px] text-slate-400">
                Fix: {u.process_fix.length > 100 ? u.process_fix.slice(0, 100) + "…" : u.process_fix}
              </p>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
