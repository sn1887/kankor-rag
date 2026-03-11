import type { SourceItem } from '@/lib/types';

export function SourcesPanel({ sources }: { sources: SourceItem[] }) {
  return (
    <aside className="rounded-3xl border border-slate-200 bg-white p-5 shadow-soft">
      <div>
        <div className="text-sm font-medium text-slate-900">Sources</div>
        <p className="mt-1 text-xs leading-6 text-slate-500">Retrieved passages shown here are the evidence base used for the answer. Corpus version and metadata are surfaced for auditability.</p>
      </div>
      <div className="mt-4 space-y-3">
        {sources.length === 0 ? <div className="rounded-2xl border border-dashed border-slate-200 bg-slate-50 p-4 text-sm text-slate-500">Select an assistant response with retrieval results to inspect its citations.</div> : null}
        {sources.map((source) => (
          <article key={source.id} className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
            <div className="flex flex-wrap items-center gap-2"><span className="rounded-full bg-blue-100 px-2.5 py-1 text-[11px] font-semibold uppercase tracking-wide text-blue-800">{source.badge}</span><span className="text-xs text-slate-500">score {source.score.toFixed(3)}</span></div>
            <h3 className="mt-3 text-sm font-semibold text-slate-900">{source.title}</h3>
            <p className="mt-2 whitespace-pre-wrap text-sm leading-6 text-slate-700">{source.snippet}</p>
            <dl className="mt-3 grid grid-cols-2 gap-2 text-xs text-slate-500">
              <div><dt className="font-semibold text-slate-700">Language</dt><dd>{source.language}</dd></div>
              <div><dt className="font-semibold text-slate-700">Subject</dt><dd>{source.subject}</dd></div>
              <div><dt className="font-semibold text-slate-700">Grade band</dt><dd>{source.gradeBand}</dd></div>
              <div><dt className="font-semibold text-slate-700">Corpus version</dt><dd>{source.corpusVersion}</dd></div>
            </dl>
          </article>
        ))}
      </div>
    </aside>
  );
}
