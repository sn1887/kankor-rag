'use client';
import clsx from 'clsx';
import ReactMarkdown from 'react-markdown';
import rehypeSanitize from 'rehype-sanitize';
import remarkGfm from 'remark-gfm';
import type { ChatMessage } from '@/lib/types';

interface MessageListProps {
  messages: ChatMessage[];
  isLoading: boolean;
  selectedMessageId: string | null;
  onSelectMessage: (id: string) => void;
}

export function MessageList({ messages, isLoading, selectedMessageId, onSelectMessage }: MessageListProps) {
  return (
    <div className="h-full overflow-y-auto px-4 py-5 md:px-6">
      <div className="mx-auto flex max-w-3xl flex-col gap-4">
        {messages.map((message) => {
          const isAssistant = message.role === 'assistant';
          const isSelected = selectedMessageId === message.id;
          return (
            <button key={message.id} type="button" onClick={() => onSelectMessage(message.id)} className={clsx('text-left transition', isAssistant ? 'self-start' : 'self-end', 'w-full max-w-[92%] rounded-3xl border px-4 py-3 md:px-5', isAssistant ? 'border-slate-200 bg-slate-50 text-slate-900' : 'border-blue-600 bg-blue-600 text-white', isSelected && isAssistant && 'ring-2 ring-blue-200')}>
              <div className="mb-2 text-[11px] font-semibold uppercase tracking-[0.16em] opacity-70">{message.role}</div>
              <div className={clsx('prose-chat text-sm leading-7 md:text-[15px]', !message.content && 'opacity-60')}>
                {isAssistant ? (
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    rehypePlugins={[rehypeSanitize]}
                    components={{
                      a: ({ children, ...props }) => (
                        <a {...props} target="_blank" rel="noreferrer noopener">
                          {children}
                        </a>
                      )
                    }}
                  >
                    {message.content || (isAssistant && isLoading ? 'Streaming response…' : '')}
                  </ReactMarkdown>
                ) : (
                  <p className="whitespace-pre-wrap">{message.content}</p>
                )}
              </div>
              {isAssistant && message.sources.length > 0 ? <div className="mt-3 flex flex-wrap gap-2 text-xs text-slate-500">{message.sources.slice(0, 3).map((source) => <span key={source.id} className="rounded-full border border-slate-200 bg-white px-2.5 py-1">{source.badge}</span>)}</div> : null}
            </button>
          );
        })}
      </div>
    </div>
  );
}
