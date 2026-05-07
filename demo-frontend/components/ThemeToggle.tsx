'use client';

import { useEffect, useState } from 'react';
import { useThemeMode } from '@/hooks/useThemeMode';

export default function ThemeToggle() {
  const { theme, setTheme } = useThemeMode();
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  if (!mounted) return <div className="w-8 h-8" />;

  const isDark = theme === 'dark';

  return (
    <button
      onClick={() => setTheme(isDark ? 'light' : 'dark')}
      aria-label={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
      className="w-8 h-8 rounded-full flex items-center justify-center shadow-md transition-transform duration-150 hover:scale-110 active:scale-95"
      style={{
        background: isDark ? '#ffffff' : '#1e3a5f',
        boxShadow: isDark
          ? '0 0 10px rgba(251,191,36,0.55), 0 2px 6px rgba(0,0,0,0.3)'
          : '0 0 10px rgba(96,165,250,0.5), 0 2px 6px rgba(0,0,0,0.35)',
      }}
    >
      {isDark ? (
        <svg viewBox="0 0 24 24" className="w-4 h-4" fill="none">
          <circle cx="12" cy="12" r="4.5" fill="#f59e0b" />
          {[0,45,90,135,180,225,270,315].map(deg => {
            const rad = (deg * Math.PI) / 180;
            return <line key={deg}
              x1={12 + 6.5*Math.cos(rad)} y1={12 + 6.5*Math.sin(rad)}
              x2={12 + 9.5*Math.cos(rad)} y2={12 + 9.5*Math.sin(rad)}
              stroke="#f59e0b" strokeWidth="1.8" strokeLinecap="round" />;
          })}
        </svg>
      ) : (
        <svg viewBox="0 0 24 24" className="w-4 h-4" fill="none">
          <path d="M21 12.79A9 9 0 1 1 11.21 3a7 7 0 0 0 9.79 9.79z" fill="white" />
        </svg>
      )}
    </button>
  );
}
