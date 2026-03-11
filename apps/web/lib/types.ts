export type Role = 'user' | 'assistant' | 'system';
export interface SourceItem {
  id: string;
  badge: string;
  title: string;
  snippet: string;
  score: number;
  subject: string;
  language: string;
  gradeBand: string;
  corpusVersion: string;
}
export interface ChatMessage {
  id: string;
  role: Role;
  content: string;
  sources: SourceItem[];
}
