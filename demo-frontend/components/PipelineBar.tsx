'use client';

import { useEffect, useState } from 'react';
import { AnalysisResult } from '@/hooks/useAnalysis';

interface Props {
  isLoading: boolean;
  result: AnalysisResult | null;
}

const LAYERS = [
  { id: 'L1', label: 'ETL Rules',    key: 'L1_rules',      cssVar: '--l1' },
  { id: 'L2', label: 'FP-Growth',    key: 'L2_fpgrowth',   cssVar: '--l2' },
  { id: 'L3', label: 'DeBERTa NLP',  key: 'L3_deberta',    cssVar: '--l3' },
  { id: 'L4', label: 'Clustering',   key: 'L4_clustering', cssVar: '--l4' },
  { id: 'L5', label: 'Ensemble',     key: 'L5_ensemble',   cssVar: '--l5' },
];

const lc = (v: string) => `var(${v})`;

// Each layer lights up at this offset (ms) after loading starts.
// 5 layers spread across ~1.6 s so they all finish before the ~1.8 s backend delay.
const STEP_DELAYS = [0, 340, 700, 1060, 1420];

export default function PipelineBar({ isLoading, result }: Props) {
  // activeStep: index of the layer currently being "processed" (0-based, -1 = idle)
  const [activeStep, setActiveStep] = useState(-1);
  // doneUpTo: layers completed so far (0 = none done)
  const [doneUpTo, setDoneUpTo] = useState(0);

  useEffect(() => {
    if (isLoading) {
      setActiveStep(0);
      setDoneUpTo(0);

      const timers: ReturnType<typeof setTimeout>[] = [];

      STEP_DELAYS.forEach((delay, i) => {
        // Ball arrives at layer i
        timers.push(setTimeout(() => setActiveStep(i), delay));
        // Layer i completes 280 ms after arriving
        timers.push(setTimeout(() => setDoneUpTo(i + 1), delay + 280));
      });

      return () => timers.forEach(clearTimeout);
    }

    if (result) {
      setActiveStep(-1);
      setDoneUpTo(LAYERS.length);
      return;
    }

    // Reset (e.g. after hitting Reset button)
    setActiveStep(-1);
    setDoneUpTo(0);
  }, [isLoading, result]);

  const statusLabel = () => {
    if (result)   return <span className="text-xs text-emerald-400 font-medium">✓ All layers complete</span>;
    if (isLoading && activeStep >= 0)
      return (
        <span className="text-xs font-medium" style={{ color: lc(LAYERS[activeStep].cssVar) }}>
          Running {LAYERS[activeStep].id}: {LAYERS[activeStep].label}…
        </span>
      );
    return null;
  };

  return (
    <div className="px-4 pb-3 max-w-4xl mx-auto w-full">
      <div className="glass rounded-2xl p-3 pb-4">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-medium tracking-widest uppercase" style={{ color: 'var(--text-faint)' }}>
            Detection Pipeline
          </span>
          {statusLabel()}
        </div>

        {/* Track row */}
        <div className="relative flex items-center">
          {LAYERS.map((layer, idx) => {
            const isDone   = doneUpTo > idx;
            const isActive = activeStep === idx && isLoading;
            const isIdle   = !isDone && !isActive;

            return (
              <div key={layer.id} className="flex items-center flex-1 min-w-0">
                {/* ── Node ── */}
                <div className="flex flex-col items-center gap-2 flex-1">
                  {/* Outer pulse ring when active */}
                  <div className="relative flex items-center justify-center">
                    {isActive && (
                      <span
                        className="absolute w-12 h-12 rounded-full animate-ping opacity-25"
                        style={{ background: lc(layer.cssVar) }}
                      />
                    )}
                    <div
                      className="relative w-9 h-9 rounded-full flex items-center justify-center text-xs font-bold border-2 z-10"
                      style={{
                        borderColor: isDone || isActive ? lc(layer.cssVar) : 'rgba(148,163,184,0.15)',
                        background : isDone ? `color-mix(in srgb, ${lc(layer.cssVar)} 16%, transparent)`
                                   : isActive ? `color-mix(in srgb, ${lc(layer.cssVar)} 10%, transparent)`
                                   : 'var(--glass-bg)',
                        color      : isDone || isActive ? lc(layer.cssVar) : 'var(--text-faint)',
                        boxShadow  : isDone  ? `0 0 14px color-mix(in srgb, ${lc(layer.cssVar)} 34%, transparent)`
                                   : isActive ? `0 0 20px color-mix(in srgb, ${lc(layer.cssVar)} 53%, transparent)`
                                   : 'none',
                        transform  : isActive ? 'scale(1.15)' : 'scale(1)',
                        transition : 'all 0.3s cubic-bezier(0.4,0,0.2,1)',
                      }}
                    >
                      {isDone ? (
                        <svg viewBox="0 0 12 12" className="w-4 h-4" fill="none">
                          <path d="M2 6l3 3 5-5" stroke={lc(layer.cssVar)} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                        </svg>
                      ) : layer.id}
                    </div>
                  </div>

                  {/* Score bar */}
                  <div className="w-full max-w-[72px] layer-bar-track h-1.5 transition-opacity duration-500"
                    style={{ opacity: result ? 1 : 0 }}>
                    <div className="layer-bar-fill h-full" style={{
                      width: result
                        ? `${Math.min(result.layer_scores[layer.key as keyof typeof result.layer_scores] * 200, 100)}%`
                        : '0%',
                      background: `linear-gradient(90deg, color-mix(in srgb, ${lc(layer.cssVar)} 40%, transparent), ${lc(layer.cssVar)})`,
                      transition: 'width 1.2s cubic-bezier(0.4,0,0.2,1)',
                    }} />
                  </div>

                  <span className="text-[10px] font-medium text-center leading-tight" style={{ color: 'var(--text-faint)' }}>
                    {layer.label}
                  </span>
                </div>

                {/* ── Connector ── */}
                {idx < LAYERS.length - 1 && (
                  <div
                    className="relative h-px flex-shrink-0 mx-0.5 overflow-hidden"
                    style={{ width: 28, marginBottom: result ? 36 : 42 }}
                  >
                    {/* Track */}
                    <div className="absolute inset-0 rounded-full" style={{ background: 'var(--bar-track)' }} />

                    {/* Fill */}
                    <div className="absolute inset-y-0 left-0 rounded-full" style={{
                      width     : doneUpTo > idx ? '100%' : '0%',
                      background: `linear-gradient(90deg, ${lc(layer.cssVar)}, ${lc(LAYERS[idx+1].cssVar)})`,
                      boxShadow : doneUpTo > idx ? `0 0 6px ${lc(layer.cssVar)}` : 'none',
                      transition: 'width 0.35s cubic-bezier(0.4,0,0.2,1)',
                    }} />

                    {/* Travelling ball */}
                    {activeStep === idx + 1 && isLoading && (
                      <div className="absolute top-1/2 -translate-y-1/2 w-2.5 h-2.5 rounded-full" style={{
                        right    : -5,
                        background: lc(LAYERS[idx+1].cssVar),
                        boxShadow : `0 0 8px ${lc(LAYERS[idx+1].cssVar)}`,
                        animation : 'travelBall 0.34s cubic-bezier(0.4,0,0.2,1) forwards',
                      }} />
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      <style>{`
        @keyframes travelBall {
          from { right: 100%; opacity: 0; }
          to   { right: -5px; opacity: 1; }
        }
      `}</style>
    </div>
  );
}
