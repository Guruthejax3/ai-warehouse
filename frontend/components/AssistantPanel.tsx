"use client";

// Assistant chat: every answer is produced by the backend from structured
// event-log queries (never free-generated facts).

import { FormEvent, useState } from "react";
import { askAssistant } from "@/lib/api";

interface Message {
  role: "user" | "assistant";
  text: string;
}

const SUGGESTIONS = [
  "How many critical events have been detected?",
  "What are the most common risky behaviors in bay 7?",
  "Show me recent product dropped events.",
];

export default function AssistantPanel() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function send(text: string) {
    if (!text.trim() || busy) return;
    setMessages((m) => [...m, { role: "user", text }]);
    setInput("");
    setBusy(true);
    setError(null);
    try {
      const reply = await askAssistant(text.trim());
      setMessages((m) => [...m, { role: "assistant", text: reply }]);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Assistant call failed");
    } finally {
      setBusy(false);
    }
  }

  function onSubmit(ev: FormEvent) {
    ev.preventDefault();
    void send(input);
  }

  return (
    <section className="flex min-h-0 flex-1 flex-col rounded-xl border border-slate-200 bg-white shadow-sm">
      <header className="border-b border-slate-100 px-4 py-3">
        <h2 className="text-sm font-semibold text-slate-700">Field assistant</h2>
        <p className="text-[11px] text-slate-400">
          Answers sourced only from the event log via tool calls.
        </p>
      </header>

      <div className="flex-1 space-y-3 overflow-y-auto px-4 py-3" style={{ maxHeight: 320 }}>
        {messages.length === 0 && (
          <div className="space-y-2">
            <p className="text-xs text-slate-400">Try asking:</p>
            {SUGGESTIONS.map((s) => (
              <button
                key={s}
                onClick={() => void send(s)}
                className="block w-full rounded-lg border border-indigo-100 bg-indigo-50 px-3 py-2 text-left text-xs text-indigo-700 hover:bg-indigo-100"
              >
                {s}
              </button>
            ))}
          </div>
        )}
        {messages.map((m, i) => (
          <div
            key={i}
            className={`max-w-[90%] whitespace-pre-wrap rounded-lg px-3 py-2 text-sm ${
              m.role === "user"
                ? "ml-auto bg-indigo-600 text-white"
                : "bg-slate-100 text-slate-800"
            }`}
          >
            {m.text}
          </div>
        ))}
        {busy && <p className="text-xs text-slate-400">Assistant is thinking…</p>}
        {error && <p className="text-xs text-red-600">{error}</p>}
      </div>

      <form onSubmit={onSubmit} className="flex gap-2 border-t border-slate-100 p-3">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about the event log…"
          className="min-w-0 flex-1 rounded-lg border border-slate-200 px-3 py-2 text-sm outline-none focus:border-indigo-400"
        />
        <button
          type="submit"
          disabled={busy}
          className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700 disabled:opacity-50"
        >
          Send
        </button>
      </form>
    </section>
  );
}