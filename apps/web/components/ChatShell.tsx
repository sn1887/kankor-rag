'use client';

import { useMemo, useState } from 'react';
import { PromptBox } from '@/components/PromptBox';
import { MessageList } from '@/components/MessageList';
import { SourcesPanel } from '@/components/SourcesPanel';
import { consumeEventStream } from '@/lib/uiStream';
import type { ChatMessage, SourceItem } from '@/lib/types';

const starterPrompts = [
  'Explain how to solve a simple algebra equation with steps.',
  'د درجه دوم معادلو لنډه تشریح راکړه.',
  'در مورد انرژی جنبشی با مثال ساده توضیح بده.'
];

const makeId = (prefix: string) => `${prefix}-${Math.random().toString(36).slice(2, 10)}`;

export function ChatShell() {
  const [messages, setMessages] = useState<ChatMessage[]>([{ id: makeId('assistant'), role: 'assistant', content: 'Ask about Kankor subjects, concepts, worked examples, or study strategy. I will retrieve supporting passages first and then answer with grounded explanations.', sources: [] }]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedMessageId, setSelectedMessageId] = useState<string | null>(null);

  const selectedSources = useMemo(() => {
    const selected = messages.find((item) => item.id === selectedMessageId);
    if (selected?.sources?.length) return selected.sources;
    const latest = [...messages].reverse().find((item) => item.role === 'assistant' && item.sources.length > 0);
    return latest?.sources ?? [];
  }, [messages, selectedMessageId]);

  async function sendMessage(input: string) {
    const trimmed = input.trim();
    if (!trimmed || isLoading) return;
    const userMessage: ChatMessage = { id: makeId('user'), role: 'user', content: trimmed, sources: [] };
    const assistantId = makeId('assistant');
    const assistantMessage: ChatMessage = { id: assistantId, role: 'assistant', content: '', sources: [] };
    const nextMessages = [...messages, userMessage, assistantMessage];
    setMessages(nextMessages);
    setSelectedMessageId(assistantId);
    setIsLoading(true);
    setError(null);

    try {
      const requestMessages = nextMessages
        .map(({ role, content }) => ({ role, content: content.trim() }))
        .filter(({ content }) => content.length > 0);

      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: requestMessages })
      });
      if (!response.ok || !response.body) throw new Error((await response.text()) || 'Invalid response from chat route.');
      await consumeEventStream(response, {
        onSources: (sources: SourceItem[]) => setMessages((current) => current.map((item) => item.id === assistantId ? { ...item, sources } : item)),
        onDelta: (delta: string) => setMessages((current) => current.map((item) => item.id === assistantId ? { ...item, content: `${item.content}${delta}` } : item)),
        onError: (message: string) => { throw new Error(message); }
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown stream error.';
      setError(message);
      setMessages((current) => current.map((item) => item.id === assistantId && !item.content ? { ...item, content: 'I could not complete the response stream. Please try again or switch to demo mode while validating the stack.' } : item));
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="mx-auto flex min-h-screen max-w-7xl flex-col gap-6 px-4 py-6 lg:px-8">
      <section className="rounded-3xl border border-white/70 bg-white/80 p-6 shadow-soft backdrop-blur">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
          <div className="max-w-3xl">
            <span className="inline-flex rounded-full border border-blue-200 bg-blue-50 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-blue-700">Modular RAG / Hugging Face Free Tier</span>
            <h1 className="mt-3 text-3xl font-semibold tracking-tight text-ink md:text-4xl">Kankor exam assistant template</h1>
            <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-600 md:text-base">Source-first retrieval, multilingual-friendly corpus design, offline index builds, and a chat UI shaped for low-bandwidth environments.</p>
          </div>
          <div className="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-600"><div className="font-medium text-slate-900">Pilot defaults</div><div>Qwen3.5-2B · multilingual-e5-small · FAISS</div></div>
        </div>
      </section>

      <section className="grid flex-1 gap-6 lg:grid-cols-[minmax(0,1.6fr)_360px]">
        <div className="flex min-h-[70vh] flex-col rounded-3xl border border-slate-200 bg-white shadow-soft">
          <div className="border-b border-slate-200 px-5 py-4">
            <div className="text-sm font-medium text-slate-900">Chat</div>
            <div className="mt-1 text-xs text-slate-500">Answers stream as retrieved context is grounded into a response. Click an assistant turn to inspect its sources.</div>
          </div>
          <div className="flex-1 overflow-hidden"><MessageList messages={messages} isLoading={isLoading} selectedMessageId={selectedMessageId} onSelectMessage={setSelectedMessageId} /></div>
          <div className="border-t border-slate-200 p-4">
            <div className="mb-3 flex flex-wrap gap-2">{starterPrompts.map((prompt) => <button key={prompt} className="rounded-full border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-600 transition hover:border-blue-200 hover:bg-blue-50 hover:text-blue-700" onClick={() => void sendMessage(prompt)} disabled={isLoading}>{prompt}</button>)}</div>
            <PromptBox onSubmit={sendMessage} disabled={isLoading} />
            {error ? <p className="mt-3 text-sm text-rose-600">{error}</p> : null}
          </div>
        </div>
        <SourcesPanel sources={selectedSources} />
      </section>
    </main>
  );
}



