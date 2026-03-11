import type { SourceItem } from '@/lib/types';

interface StreamHandlers {
  onSources?: (sources: SourceItem[]) => void;
  onDelta?: (delta: string) => void;
  onError?: (message: string) => void;
}

function parseEventBlock(block: string) {
  const lines = block.split(/\r?\n/);
  let event = 'message';
  const dataLines: string[] = [];

  for (const raw of lines) {
    const line = raw.trimEnd();
    if (line.startsWith('event:')) {
      event = line.slice(6).trim();
    } else if (line.startsWith('data:')) {
      dataLines.push(line.slice(5).trimStart());
    }
  }

  return { event, data: dataLines.join('\n') };
}

export async function consumeEventStream(
  response: Response,
  handlers: StreamHandlers
) {
  if (!response.body) {
    throw new Error('The response did not include a readable body.');
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split(/\r?\n\r?\n/);
    buffer = parts.pop() ?? '';

    for (const part of parts) {
      if (!part.trim()) continue;

      const { event, data } = parseEventBlock(part);

      switch (event) {
        case 'sources':
          handlers.onSources?.(JSON.parse(data) as SourceItem[]);
          break;
        case 'delta':
          handlers.onDelta?.(JSON.parse(data).text as string);
          break;
        case 'error':
          handlers.onError?.(JSON.parse(data).message as string);
          break;
        case 'done':
          return;
      }
    }
  }
}
