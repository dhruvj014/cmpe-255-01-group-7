'use client';

import { useThemeMode } from '@/hooks/useThemeMode';

const BLOB_COLORS = {
  dark:  { b1: '#f59e0b', b2: '#e11d48', b3: '#f97316' },
  light: { b1: '#38bdf8', b2: '#3b82f6', b3: '#60a5fa' },
};

export default function BackgroundBlobs() {
  const { theme } = useThemeMode();
  const c = BLOB_COLORS[theme];

  const blob = (color: string) =>
    `radial-gradient(circle, ${color}, transparent 70%)`;

  return (
    <div className="bg-mesh" aria-hidden>
      <div className="blob blob-1" style={{ background: blob(c.b1) }} />
      <div className="blob blob-2" style={{ background: blob(c.b2) }} />
      <div className="blob blob-3" style={{ background: blob(c.b3) }} />
    </div>
  );
}
