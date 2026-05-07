'use client';

import { useState } from 'react';
import { ReviewData } from '@/hooks/useAnalysis';
import { ToastType } from '@/hooks/useToast';

const PRESETS = [
  {
    label: '🚨 Obvious Spam',
    data: { text: 'Best product ever!! ABSOLUTELY LOVE IT! Best best best!! Must buy now!!!',
            rating: 5, tenure_days: 3, review_count: 1, seller_concentration: 1.0, burst_score: 8 },
  },
  {
    label: '✅ Legitimate',
    data: { text: 'Decent burger, a bit overpriced but the service was friendly and the wait was reasonable.',
            rating: 3, tenure_days: 730, review_count: 47, seller_concentration: 0.1, burst_score: 1 },
  },
  {
    label: '⚠️ Borderline',
    data: { text: 'Terrible experience! Will never come back. Complete waste of money.',
            rating: 1, tenure_days: 25, review_count: 2, seller_concentration: 0.9, burst_score: 3 },
  },
];

interface Props {
  initialData: ReviewData;
  isLoading: boolean;
  onSubmit: (data: ReviewData) => void;
  showToast: (message: string, type?: ToastType) => void;
}

function validate(data: ReviewData): string | null {
  if (!data.text.trim())             return 'Review text is required.';
  if (data.text.trim().length < 10)  return `Review is too short — needs at least 10 characters (${data.text.trim().length}/10).`;
  if (data.rating < 1 || data.rating > 5) return 'Please select a star rating between 1 and 5.';
  if (data.tenure_days < 0)          return 'Account tenure cannot be negative.';
  if (data.review_count < 1)         return 'Review count must be at least 1.';
  if (data.seller_concentration < 0 || data.seller_concentration > 1)
                                     return 'Seller concentration must be between 0.0 and 1.0.';
  if (data.burst_score < 0)          return 'Burst score cannot be negative.';
  return null;
}

// Semantic concentration bar — uses CSS vars so it adapts to theme
function concBar(val: number) {
  if (val > 0.7) return `linear-gradient(90deg, var(--l4), var(--l5))`;
  if (val > 0.4) return `linear-gradient(90deg, var(--l2), var(--l3))`;
  return 'linear-gradient(90deg, #10b981, #34d399)';
}

export default function InputForm({ isLoading, onSubmit, showToast }: Props) {
  const [data, setData] = useState<ReviewData>({
    text: '', rating: 5, tenure_days: 0, review_count: 1, seller_concentration: 1.0, burst_score: 1,
  });

  const update = <K extends keyof ReviewData>(key: K, value: ReviewData[K]) =>
    setData(prev => ({ ...prev, [key]: value }));

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const err = validate(data);
    if (err) { showToast(err, 'warning'); return; }
    onSubmit(data);
  };

  return (
    <div className="glass rounded-2xl p-4 flex flex-col gap-3 h-full overflow-y-auto">
      {/* Title */}
      <div className="shrink-0">
        <h2 className="text-base font-bold" style={{ color: 'var(--text)' }}>Review Input</h2>
        <p className="text-xs" style={{ color: 'var(--text-faint)' }}>Fill in the review text and behavioral signals.</p>
      </div>

      {/* Quick-fill presets */}
      <div>
        <p className="text-xs uppercase tracking-widest font-medium mb-2" style={{ color: 'var(--text-faint)' }}>Quick Fill</p>
        <div className="flex flex-wrap gap-2">
          {PRESETS.map(p => (
            <button
              key={p.label}
              type="button"
              onClick={() => setData(p.data)}
              className="px-3 py-1.5 text-xs font-medium rounded-lg transition-all"
              style={{
                background  : 'var(--glass-bg)',
                border      : '1px solid var(--glass-border)',
                color       : 'var(--text-muted)',
              }}
              onMouseEnter={e => (e.currentTarget.style.borderColor = 'var(--l3)')}
              onMouseLeave={e => (e.currentTarget.style.borderColor = 'var(--glass-border)')}
            >
              {p.label}
            </button>
          ))}
        </div>
      </div>

      <form onSubmit={handleSubmit} noValidate className="flex flex-col gap-3">
        {/* Review text */}
        <div>
          <label className="block text-xs font-semibold uppercase tracking-widest mb-2" style={{ color: 'var(--text-muted)' }}>
            Review Text
          </label>
          <textarea
            rows={3}
            className="glass-input w-full rounded-xl px-4 py-3 text-sm resize-none"
            placeholder="Write the review text here (min 10 characters)…"
            value={data.text}
            onChange={e => update('text', e.target.value)}
          />
          <div className="flex justify-end mt-1">
            <span className="text-xs" style={{ color: 'var(--text-faint)' }}>{data.text.length} chars</span>
          </div>
        </div>

        {/* Star rating */}
        <div>
          <label className="block text-xs font-semibold uppercase tracking-widest mb-2" style={{ color: 'var(--text-muted)' }}>
            Star Rating
          </label>
          <div className="flex gap-2 items-center">
            {[1, 2, 3, 4, 5].map(n => (
              <button
                key={n}
                type="button"
                onClick={() => update('rating', n)}
                className="star text-2xl transition-all"
                style={{ color: n <= data.rating ? 'var(--l2)' : 'var(--bar-track)' }}
              >
                ★
              </button>
            ))}
            <span className="ml-2 text-sm self-center" style={{ color: 'var(--text-muted)' }}>
              {data.rating} / 5
            </span>
          </div>
        </div>

        {/* Behavioral signals */}
        <div className="grid grid-cols-2 gap-3">
          <FieldInput label="Account Tenure"     hint="Days since account creation"
            type="number" min={0}    value={data.tenure_days}          suffix="days"    onChange={v => update('tenure_days', Number(v))} />
          <FieldInput label="Total Reviews"      hint="Lifetime review count"
            type="number" min={1}    value={data.review_count}         suffix="reviews" onChange={v => update('review_count', Number(v))} />
          <FieldInput
            label="Seller Concentration"
            hint="% of this user's reviews targeting ONE business"
            tooltip="A real customer reviews many different places. A fake reviewer is paid to flood ONE business — so nearly all their reviews point at the same seller. 0.0 = many businesses (normal). 1.0 = one business only (suspicious)."
            type="number" min={0} max={1} step={0.05}
            value={data.seller_concentration} suffix=""
            onChange={v => update('seller_concentration', Number(v))}
            renderExtra={
              <div className="mt-1.5">
                <div className="h-1.5 rounded-full overflow-hidden" style={{ background: 'var(--bar-track)' }}>
                  <div
                    className="h-full rounded-full transition-all duration-300"
                    style={{ width: `${data.seller_concentration * 100}%`, background: concBar(data.seller_concentration) }}
                  />
                </div>
                <div className="flex justify-between text-[10px] mt-0.5 px-0.5" style={{ color: 'var(--text-faint)' }}>
                  <span>0.0 = Many sellers</span>
                  <span>1.0 = One seller</span>
                </div>
              </div>
            }
          />
          <FieldInput label="Burst Score" hint="Max reviews in a 7-day window"
            type="number" min={0}    value={data.burst_score}          suffix="/ week"  onChange={v => update('burst_score', Number(v))} />
        </div>

        {/* Submit */}
        <button
          type="submit"
          disabled={isLoading}
          className="btn-primary w-full py-3 rounded-xl font-semibold flex items-center justify-center gap-3 text-sm mt-1"
        >
          {isLoading ? (
            <>
              <svg className="animate-spin-custom w-5 h-5" viewBox="0 0 24 24" fill="none">
                <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" strokeOpacity="0.3" />
                <path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
              </svg>
              Analyzing 5 Layers…
            </>
          ) : (
            <><span>⚡</span> Analyze Review Pipeline</>
          )}
        </button>
      </form>
    </div>
  );
}

interface FieldInputProps {
  label: string; hint: string; tooltip?: string;
  type: string; min?: number; max?: number; step?: number;
  value: number; suffix: string;
  onChange: (v: string) => void;
  renderExtra?: React.ReactNode;
}

function FieldInput({ label, hint, tooltip, type, min, max, step, value, suffix, onChange, renderExtra }: FieldInputProps) {
  return (
    <div>
      <div className="flex items-center gap-1.5 mb-1">
        <label className="block text-xs font-semibold uppercase tracking-widest" style={{ color: 'var(--text-muted)' }}>
          {label}
        </label>
        {tooltip && (
          <div className="relative group">
            <span className="w-3.5 h-3.5 rounded-full text-[9px] font-bold flex items-center justify-center cursor-help select-none"
              style={{ background: 'var(--bar-track)', border: '1px solid var(--glass-border)', color: 'var(--text-muted)' }}>
              ?
            </span>
            <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-56 rounded-lg p-2.5 text-[11px] leading-relaxed shadow-xl z-50 pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity duration-150"
              style={{ background: 'var(--glass-bg)', border: '1px solid var(--glass-border)', color: 'var(--text-muted)', backdropFilter: 'blur(16px)' }}>
              {tooltip}
              <div className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent" style={{ borderTopColor: 'var(--glass-border)' }} />
            </div>
          </div>
        )}
      </div>
      <p className="text-[10px] mb-1" style={{ color: 'var(--text-faint)' }}>{hint}</p>
      <div className="relative">
        <input
          type={type} min={min} max={max} step={step} required
          className="glass-input w-full rounded-lg px-3 py-2 text-sm pr-16"
          value={value}
          onChange={e => onChange(e.target.value)}
        />
        {suffix && (
          <span className="absolute right-3 top-1/2 -translate-y-1/2 text-xs pointer-events-none" style={{ color: 'var(--text-faint)' }}>
            {suffix}
          </span>
        )}
      </div>
      {renderExtra}
    </div>
  );
}
