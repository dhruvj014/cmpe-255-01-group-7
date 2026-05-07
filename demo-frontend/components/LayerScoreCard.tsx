'use client';

import { LayerScores } from '@/hooks/useAnalysis';

interface Props { scores: LayerScores; }

const LAYERS: { key: keyof LayerScores; shortLabel: string; cssVar: string }[] = [
  { key: 'L1_rules',      shortLabel: 'L1', cssVar: '--l1' },
  { key: 'L2_fpgrowth',   shortLabel: 'L2', cssVar: '--l2' },
  { key: 'L3_deberta',    shortLabel: 'L3', cssVar: '--l3' },
  { key: 'L4_clustering', shortLabel: 'L4', cssVar: '--l4' },
  { key: 'L5_ensemble',   shortLabel: 'L5', cssVar: '--l5' },
];
const lc = (v: string) => `var(${v})`;

export default function LayerScoreCard({ scores }: Props) {
  return (
    <div className="glass rounded-xl p-4 flex flex-col gap-2">
      <span className="text-xs font-semibold uppercase tracking-widest mb-1" style={{ color: 'var(--text-faint)' }}>
        Layer Scores
      </span>
      {LAYERS.map((layer, i) => {
        const raw = scores[layer.key] ?? 0;
        const pct = layer.key === 'L5_ensemble'
          ? Math.min(raw * 100, 100)
          : Math.min(raw * 200, 100);
        return (
          <div key={layer.key} className="flex items-center gap-2" style={{ animationDelay: `${i * 60}ms` }}>
            <span className="text-[10px] font-bold w-5 shrink-0" style={{ color: lc(layer.cssVar) }}>
              {layer.shortLabel}
            </span>
            <div className="flex-1 layer-bar-track h-2">
              <div
                className="layer-bar-fill h-full"
                style={{
                  width     : `${pct}%`,
                  background: `linear-gradient(90deg, color-mix(in srgb, ${lc(layer.cssVar)} 40%, transparent), ${lc(layer.cssVar)})`,
                  boxShadow : pct > 30 ? `0 0 6px color-mix(in srgb, ${lc(layer.cssVar)} 35%, transparent)` : 'none',
                }}
              />
            </div>
            <span className="text-[10px] w-8 text-right shrink-0" style={{ color: 'var(--text-faint)' }}>
              {(raw * 100).toFixed(0)}%
            </span>
          </div>
        );
      })}
    </div>
  );
}
