import { NextRequest } from 'next/server';
export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function POST(request: NextRequest) {
  const backendUrl = process.env.BACKEND_URL ?? 'http://127.0.0.1:8000';
  const backendApiKey = process.env.BACKEND_API_KEY?.trim();
  const payload = await request.json();
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    Accept: 'text/event-stream'
  };
  if (backendApiKey) {
    headers.Authorization = `Bearer ${backendApiKey}`;
  }
  const response = await fetch(`${backendUrl}/v1/chat/stream`, {
    method: 'POST',
    headers,
    body: JSON.stringify(payload),
    cache: 'no-store'
  });

  return new Response(response.body, {
    status: response.status,
    headers: {
      'Content-Type': 'text/event-stream; charset=utf-8',
      'Cache-Control': 'no-cache, no-transform',
      Connection: 'keep-alive'
    }
  });
}
