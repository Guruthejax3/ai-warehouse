import LiveEventFeed from "@/components/LiveEventFeed";
import BayHeatmap from "@/components/BayHeatmap";
import BehaviorTrends from "@/components/BehaviorTrends";
import SafetyScorecard from "@/components/SafetyScorecard";
import AssistantPanel from "@/components/AssistantPanel";

export default function Page() {
  return (
    <main className="mx-auto flex h-full w-full max-w-7xl flex-col gap-4 p-4">
      <header className="flex items-baseline gap-3">
        <h1 className="text-xl font-bold text-slate-900">ReplayTwin</h1>
        <span className="text-sm text-slate-400">
          AI field intelligence · warehouse handling video
        </span>
      </header>

      <SafetyScorecard />

      <div className="grid flex-1 grid-cols-1 gap-4 xl:grid-cols-3">
        <div className="flex flex-col gap-4 xl:col-span-2">
          <LiveEventFeed />
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <BayHeatmap />
            <BehaviorTrends />
          </div>
        </div>
        <div className="flex min-h-96 flex-col">
          <AssistantPanel />
        </div>
      </div>
    </main>
  );
}