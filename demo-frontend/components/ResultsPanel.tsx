'use client';

import { AnalysisResult } from '@/hooks/useAnalysis';
import GaugeMeter from './GaugeMeter';
import LayerScoreCard from './LayerScoreCard';

interface Props {
  isLoading: boolean;
  error: string | null;
  result: AnalysisResult | null;
  onReset: () => void;
}

export default function ResultsPanel({ isLoading, error, result, onReset }: Props) {
  const isSpam = result?.is_spam;

  return (
    <div className="glass rounded-2xl p-4 flex flex-col gap-3 h-full overflow-y-auto">
      {/* Title row */}
      <div className="flex items-center justify-between shrink-0">
        <div>
          <h2 className="text-base font-bold" style={{ color: 'var(--text)' }}>Detection Results</h2>
          <p className="text-xs" style={{ color: 'var(--text-faint)' }}>Multi-layer pipeline output</p>
        </div>
        {result && (
          <button
            onClick={onReset}
            className="text-xs transition-colors px-2.5 py-1 rounded-lg"
            style={{ color: 'var(--text-faint)', border: '1px solid var(--glass-border)' }}
            onMouseEnter={e => (e.currentTarget.style.color = 'var(--text-muted)')}
            onMouseLeave={e => (e.currentTarget.style.color = 'var(--text-faint)')}
          >
            ↺ Reset
          </button>
        )}
      </div>

      {isLoading && <LoadingSkeleton />}

      {/* Error state */}
      {!isLoading && error && (
        <div className="animate-fade-up flex-1 flex flex-col justify-center">
          <div className="rounded-xl p-5 flex gap-4" style={{ background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.3)' }}>
            <div className="w-10 h-10 rounded-full flex items-center justify-center shrink-0 text-lg"
              style={{ background: 'rgba(239,68,68,0.15)', border: '1px solid rgba(239,68,68,0.3)', color: '#f87171' }}>
              ✕
            </div>
            <div>
              <h3 className="font-semibold mb-1" style={{ color: '#f87171' }}>Integration Error</h3>
              <p className="text-sm" style={{ color: 'rgba(248,113,113,0.75)' }}>{error}</p>
              <p className="text-xs mt-2" style={{ color: 'var(--text-faint)' }}>
                Make sure the FastAPI backend is running:{' '}
                <code className="px-1.5 py-0.5 rounded text-xs" style={{ background: 'var(--bar-track)', color: 'var(--text-muted)' }}>
                  uvicorn main:app --reload
                </code>
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Empty state */}
      {!isLoading && !error && !result && (
        <div className="flex-1 flex flex-col items-center justify-center text-center gap-3">
          <div className="relative">
            <div className="w-16 h-16 rounded-full border-2 border-dashed flex items-center justify-center text-3xl"
              style={{ borderColor: 'var(--glass-border)' }}>
              🔍
            </div>
            <div className="absolute -bottom-1 -right-1 w-6 h-6 rounded-full flex items-center justify-center text-[9px] font-bold"
              style={{ background: 'color-mix(in srgb, var(--l3) 15%, transparent)', border: '1px solid color-mix(in srgb, var(--l3) 30%, transparent)', color: 'var(--l3)' }}>
              AI
            </div>
          </div>
          <div>
            <p className="font-medium text-sm" style={{ color: 'var(--text-muted)' }}>Awaiting analysis</p>
            <p className="text-xs mt-0.5" style={{ color: 'var(--text-faint)' }}>Fill in the form and click Analyze.</p>
          </div>
        </div>
      )}

      {/* Results */}
      {!isLoading && result && (
        <div className="flex flex-col gap-3 animate-fade-up">
          {/* Verdict banner */}
          <div
            className={`rounded-xl px-4 py-3 border flex items-center gap-4 relative overflow-hidden ${
              isSpam ? 'glow-red' : 'glow-green'
            }`}
            style={{
              background: isSpam ? 'rgba(239,68,68,0.1)'  : 'rgba(16,185,129,0.1)',
              border    : isSpam ? '1px solid rgba(239,68,68,0.4)' : '1px solid rgba(16,185,129,0.4)',
            }}
          >
            <div className="absolute inset-0 flex items-center justify-center opacity-5 text-[90px] pointer-events-none select-none">
              {isSpam ? '⚠' : '✓'}
            </div>
            <span className="text-3xl shrink-0">{isSpam ? '🚨' : '✅'}</span>
            <div className="flex-1 min-w-0">
              <h3 className={`text-xl font-black tracking-wider ${isSpam ? 'gradient-text-red' : 'gradient-text-green'}`}>
                {isSpam ? 'SPAM DETECTED' : 'LEGITIMATE'}
              </h3>
              <div className="flex items-center gap-3 mt-0.5 flex-wrap">
                <TierBadge tier={result.behavioral_risk_tier} />
                <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                  L5 Confidence:{' '}
                  <span className="font-bold" style={{ color: 'var(--text)' }}>
                    {(result.spam_probability * 100).toFixed(1)}%
                  </span>
                </span>
              </div>
            </div>
          </div>

          {/* Gauge + Layer scores */}
          <div className="grid grid-cols-2 gap-3">
            <GaugeMeter probability={result.spam_probability} isSpam={isSpam ?? false} />
            <LayerScoreCard scores={result.layer_scores} />
          </div>

          {/* Top factors */}
          <div>
            <h4 className="text-xs font-semibold uppercase tracking-widest mb-2" style={{ color: 'var(--text-faint)' }}>
              Multi-Signal Factors ({result.top_factors.length})
            </h4>
            <ul className="flex flex-col gap-1.5">
              {result.top_factors.map((factor, i) => (
                <li key={i} className="factor-item text-xs" style={{ color: 'var(--text-muted)', animationDelay: `${i * 80}ms` }}>
                  <span className="shrink-0 font-bold" style={{
                    color: i === 0 ? '#f87171'
                         : i < 3  ? 'var(--l2)'
                         : 'var(--text-faint)',
                  }}>
                    {i === 0 ? '▲' : '►'}
                  </span>
                  <span>{factor}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}

function TierBadge({ tier }: { tier: string }) {
  const cls = tier === 'High' ? 'tier-high' : tier === 'Medium' ? 'tier-medium' : 'tier-low';
  return (
    <span className={`${cls} px-3 py-1 rounded-full text-xs font-bold uppercase tracking-widest`}>
      {tier} Risk
    </span>
  );
}

function LoadingSkeleton() {
  return (
    <div className="flex flex-col gap-3 flex-1">
      <div className="skeleton h-16 rounded-xl" />
      <div className="grid grid-cols-2 gap-3">
        <div className="skeleton h-28 rounded-xl" />
        <div className="skeleton h-28 rounded-xl" />
      </div>
      <div className="flex flex-col gap-2">
        <div className="skeleton h-9 rounded-lg" />
        <div className="skeleton h-9 rounded-lg" />
        <div className="skeleton h-9 rounded-lg" />
      </div>
    </div>
  );
}
