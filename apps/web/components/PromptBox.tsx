'use client';
import { useState } from 'react';

interface PromptBoxProps { onSubmit: (value: string) => Promise<void> | void; disabled?: boolean; }

export function PromptBox({ onSubmit, disabled = false }: PromptBoxProps) {
  const [value, setValue] = useState('');
  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const next = value.trim();
    if (!next || disabled) return;
    setValue('');
    await onSubmit(next);
  }
  return (
    <form onSubmit={handleSubmit} className="rounded-3xl border border-slate-200 bg-slate-50 p-3">
      <textarea value={value} onChange={(event) => setValue(event.target.value)} placeholder="Ask about mathematics, natural science, social science, languages, worked examples, or study strategy…" rows={4} disabled={disabled} className="min-h-[110px] w-full resize-none border-0 bg-transparent px-2 py-1 text-sm text-slate-900 outline-none placeholder:text-slate-400" />
      <div className="mt-3 flex items-center justify-between gap-3 px-2 pb-1">
        <p className="text-xs text-slate-500">The backend retrieves passages first and streams a grounded answer.</p>
        <button type="submit" disabled={disabled} className="rounded-full bg-slate-900 px-4 py-2 text-sm font-medium text-white transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-50">Send</button>
      </div>
    </form>
  );
}
