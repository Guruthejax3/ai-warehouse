"use client";

// Recurring behaviors: shows behaviors that appear across multiple bays,
// indicating systemic warehouse-wide issues rather than one-off incidents.

import { useEffect, useState } from "react";
import { getRecurring, RecurringBehavior } from "@/lib/api";

export default function RecurringBehaviors() {
  const [data, setData] = useState<RecurringBehavior[]>([]);

  useEffect(() => {
    getRecurring(7)
      .then((r) => setData(r.behaviors.filter((b) => b.recurring)))
      .catch(() => {});
  }, []);

  return (
    <section className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <header className="mb-3 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-slate-700">Recurring patterns</h2>
        <span className="text-[11px] text-slate-400">cross-bay detection</span>
      </header>

      {data.length === 0 ? (
        <p className="rounded-lg bg-slate-50 px-3 py-6 text-center text-sm text-slate-400">
          No recurring cross-bay patterns detected this week.
        </p>
      ) : (
        <div className="space-y-2">
          {data.map((b) => (
            <div
              key={b.behavior_class}
              className="rounded-lg border border-amber-200 bg-amber-50/60 px-3 py-2"
            >
              <div className="flex items-center justify-between gap-2">
                <span className="text-xs font-medium text-amber-800">
                  {b.behavior_class.replace(/_/g, " ")}
                </span>
                <span className="rounded-full bg-amber-100 px-2 py-0.5 text-[10px] font-semibold text-amber-700">
                  {b.occurrences}x across {b.distinct_bays} bays
                </span>
              </div>
              <div className="mt-1 flex flex-wrap gap-1">
                {b.bays.map((bay) => (
                  <span
                    key={bay}
                    className="rounded bg-amber-100 px-1.5 py-0.5 text-[10px] text-amber-600"
                  >
                    {bay}
                  </span>
                ))}
              </div>
              <p className="mt-1 text-[10px] text-amber-600/70">
                Max risk: {b.max_risk.toFixed(2)} — systemic pattern requiring process review.
              </p>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
