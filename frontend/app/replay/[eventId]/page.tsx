import Link from "next/link";
import ReplayViewer from "@/components/ReplayViewer";

export default async function ReplayPage({
  params,
}: {
  params: Promise<{ eventId: string }>;
}) {
  const { eventId } = await params;
  return (
    <main className="mx-auto flex h-full w-full max-w-4xl flex-col p-4">
      <header className="mb-3 flex items-center gap-3">
        <Link
          href="/"
          className="rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-sm text-slate-600 hover:bg-slate-50"
        >
          ← Dashboard
        </Link>
        <h1 className="text-lg font-bold text-slate-900">Event replay</h1>
      </header>
      <ReplayViewer eventId={eventId} />
      <p className="mt-3 text-xs text-slate-400">
        Red path = actual handling captured by the pipeline. Green path = the
        correct-technique exemplar for the matched behaviour class.
      </p>
    </main>
  );
}