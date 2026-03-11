import './globals.css';
import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Kankor RAG Space',
  description: "Modular exam-prep RAG assistant for Afghanistan's Kankor exam."
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
