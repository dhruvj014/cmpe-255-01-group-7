'use client';

import { Toast, ToastType } from '@/hooks/useToast';

interface Props {
  toasts: Toast[];
}

const icons: Record<ToastType, string> = {
  error:   '✕',
  warning: '⚠',
  info:    'ℹ',
};

// Inline styles so they respond to the CSS variable theme automatically
const typeStyle: Record<ToastType, { border: string; iconBg: string; iconBorder: string; iconColor: string }> = {
  error: {
    border      : 'rgba(239,68,68,0.4)',
    iconBg      : 'rgba(239,68,68,0.15)',
    iconBorder  : 'rgba(239,68,68,0.35)',
    iconColor   : '#f87171',
  },
  warning: {
    border      : 'rgba(245,158,11,0.4)',
    iconBg      : 'rgba(245,158,11,0.15)',
    iconBorder  : 'rgba(245,158,11,0.35)',
    iconColor   : 'var(--l2)',
  },
  info: {
    border      : 'rgba(14,165,233,0.4)',
    iconBg      : 'rgba(14,165,233,0.15)',
    iconBorder  : 'rgba(14,165,233,0.35)',
    iconColor   : '#38bdf8',
  },
};

export default function ToastContainer({ toasts }: Props) {
  if (toasts.length === 0) return null;

  return (
    <div className="fixed top-[75px] left-6 z-50 flex flex-col gap-3 items-start pointer-events-none">
      {toasts.map(toast => {
        const s = typeStyle[toast.type];
        return (
          <div
            key={toast.id}
            className={`
              flex items-start gap-3 px-4 py-3 rounded-xl
              max-w-xs w-72 pointer-events-auto
              ${toast.exiting ? 'toast-exit' : toast.buzzing ? 'toast-buzz' : 'toast-enter'}
            `}
            style={{
              background    : 'var(--glass-bg)',
              backdropFilter: 'blur(20px)',
              WebkitBackdropFilter: 'blur(20px)',
              border        : `1px solid ${s.border}`,
              boxShadow     : '0 8px 32px rgba(0,0,0,0.25)',
            }}
          >
            <span
              className="w-7 h-7 rounded-lg flex items-center justify-center text-sm font-bold shrink-0 mt-0.5"
              style={{
                background : s.iconBg,
                border     : `1px solid ${s.iconBorder}`,
                color      : s.iconColor,
              }}
            >
              {icons[toast.type]}
            </span>
            <p className="text-sm leading-snug" style={{ color: 'var(--text-muted)' }}>
              {toast.message}
            </p>
          </div>
        );
      })}

      <style>{`
        @keyframes toastIn {
          from { transform: translateX(calc(-100% - 24px)); opacity: 0; }
          to   { transform: translateX(0);                  opacity: 1; }
        }
        @keyframes toastOut {
          from { transform: translateX(0);                  opacity: 1; }
          to   { transform: translateX(calc(-100% - 24px)); opacity: 0; }
        }
        @keyframes toastBuzz {
          0%   { transform: translateX(0);    }
          15%  { transform: translateX(-8px); }
          30%  { transform: translateX(7px);  }
          45%  { transform: translateX(-6px); }
          60%  { transform: translateX(5px);  }
          75%  { transform: translateX(-3px); }
          90%  { transform: translateX(2px);  }
          100% { transform: translateX(0);    }
        }
        .toast-enter { animation: toastIn   0.45s cubic-bezier(0.22,1,0.36,1) both; }
        .toast-exit  { animation: toastOut  0.35s cubic-bezier(0.4,0,0.8,0)   both; }
        .toast-buzz  { animation: toastBuzz 0.5s  cubic-bezier(0.36,0.07,0.19,0.97) both; }
      `}</style>
    </div>
  );
}
