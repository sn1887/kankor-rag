import type { Config } from 'tailwindcss';
const config: Config = {
  content: ['./app/**/*.{js,ts,jsx,tsx,mdx}', './components/**/*.{js,ts,jsx,tsx,mdx}', './lib/**/*.{js,ts,jsx,tsx,mdx}'],
  theme: {
    extend: {
      colors: { ink: '#0f172a', muted: '#475569', line: '#e2e8f0', panel: '#f8fafc' },
      boxShadow: { soft: '0 12px 32px rgba(15, 23, 42, 0.08)' }
    }
  },
  plugins: []
};
export default config;
