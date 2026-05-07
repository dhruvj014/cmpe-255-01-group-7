'use client';

import { useState, useCallback, useRef } from 'react';

export type ToastType = 'error' | 'warning' | 'info';

export interface Toast {
  id: number;
  message: string;
  type: ToastType;
  exiting: boolean;
  buzzing: boolean;
}

export function useToast() {
  const [toast, setToast] = useState<Toast | null>(null);
  const counter      = useRef(0);
  const exitTimer    = useRef<ReturnType<typeof setTimeout> | null>(null);
  const removeTimer  = useRef<ReturnType<typeof setTimeout> | null>(null);
  const buzzTimer    = useRef<ReturnType<typeof setTimeout> | null>(null);

  const clearTimers = () => {
    if (exitTimer.current)   clearTimeout(exitTimer.current);
    if (removeTimer.current) clearTimeout(removeTimer.current);
    if (buzzTimer.current)   clearTimeout(buzzTimer.current);
  };

  const show = useCallback((message: string, type: ToastType = 'error') => {
    setToast(current => {
      // A toast is already visible — buzz it with the new message
      if (current && !current.exiting) {
        clearTimers();

        // Trigger buzz
        const buzzed: Toast = { ...current, message, type, buzzing: true };

        // Remove buzz flag after animation
        buzzTimer.current = setTimeout(() => {
          setToast(t => t ? { ...t, buzzing: false } : t);
        }, 500);

        // Restart exit timer
        exitTimer.current = setTimeout(() => {
          setToast(t => t ? { ...t, exiting: true } : t);
        }, 2400);
        removeTimer.current = setTimeout(() => {
          setToast(null);
        }, 2750);

        return buzzed;
      }

      // No toast visible — create a fresh one
      const id = ++counter.current;
      exitTimer.current = setTimeout(() => {
        setToast(t => t ? { ...t, exiting: true } : t);
      }, 2400);
      removeTimer.current = setTimeout(() => {
        setToast(null);
      }, 2750);

      return { id, message, type, exiting: false, buzzing: false };
    });
  }, []);

  // Wrap in array for ToastContainer compatibility
  const toasts: Toast[] = toast ? [toast] : [];

  return { toasts, show };
}
