'use client';

interface Props {
  probability: number;
  isSpam: boolean;
}

export default function GaugeMeter({ probability, isSpam }: Props) {
  const pct = Math.min(Math.max(probability, 0), 1);

  const r  = 58;
  const cx = 80;
  const cy = 74;

  const circumference = Math.PI * r;
  const strokeOffset  = circumference * (1 - pct);

  const color = isSpam
    ? pct > 0.7 ? '#ef4444' : '#f97316'
    : '#10b981';

  const needleAngle = 180 - pct * 180;
  const needleRad   = (needleAngle * Math.PI) / 180;
  const needleLen   = 46;
  const nx = cx + needleLen * Math.cos(needleRad);
  const ny = cy - needleLen * Math.sin(needleRad);

  const arcLeft  = cx - r;
  const arcRight = cx + r;

  return (
    <div className="glass rounded-xl p-4 flex flex-col items-center">
      <span className="text-xs font-semibold uppercase tracking-widest mb-1" style={{ color: 'var(--text-faint)' }}>
        Spam Score
      </span>

      <svg width="100%" viewBox="0 0 160 116" className="overflow-visible max-w-[180px]">
        {/* Track arc */}
        <path
          d={`M ${arcLeft} ${cy} A ${r} ${r} 0 0 1 ${arcRight} ${cy}`}
          fill="none"
          style={{ stroke: 'var(--bar-track)' }}
          strokeWidth="10"
          strokeLinecap="round"
        />

        {/* Fill arc */}
        <path
          d={`M ${arcLeft} ${cy} A ${r} ${r} 0 0 1 ${arcRight} ${cy}`}
          fill="none"
          stroke={color}
          strokeWidth="10"
          strokeLinecap="round"
          strokeDasharray={`${circumference}`}
          strokeDashoffset={`${strokeOffset}`}
          style={{
            transition: 'stroke-dashoffset 1s cubic-bezier(0.4,0,0.2,1)',
            filter: `drop-shadow(0 0 8px ${color}99)`,
          }}
        />

        {/* Tick marks */}
        {[0, 0.25, 0.5, 0.75, 1].map(t => {
          const a     = (180 - t * 180) * (Math.PI / 180);
          const inner = r - 7;
          const outer = r + 4;
          return (
            <line
              key={t}
              x1={cx + inner * Math.cos(a)} y1={cy - inner * Math.sin(a)}
              x2={cx + outer * Math.cos(a)} y2={cy - outer * Math.sin(a)}
              style={{ stroke: 'var(--glass-border)' }}
              strokeWidth="1.5"
            />
          );
        })}

        {/* Needle */}
        <line
          x1={cx} y1={cy} x2={nx} y2={ny}
          style={{ stroke: 'var(--text)', transition: 'x2 1s cubic-bezier(0.4,0,0.2,1), y2 1s cubic-bezier(0.4,0,0.2,1)' }}
          strokeWidth="2.5"
          strokeLinecap="round"
        />

        {/* Pivot */}
        <circle cx={cx} cy={cy} r="6"
          style={{ fill: 'var(--glass-bg)', stroke: 'var(--glass-border)' }}
          strokeWidth="1.5"
        />
        <circle cx={cx} cy={cy} r="3" fill={color} style={{ transition: 'fill 0.5s' }} />

        {/* Percentage */}
        <text
          x={cx} y={cy + 32}
          textAnchor="middle"
          fontSize="20" fontWeight="900"
          fill={color}
          style={{
            fontFamily: 'system-ui, -apple-system, sans-serif',
            filter: `drop-shadow(0 0 8px ${color}66)`,
            transition: 'fill 0.5s',
          }}
        >
          {(pct * 100).toFixed(0)}%
        </text>

        {/* Safe / Spam axis labels */}
        <text x={arcLeft} y={cy + 18} textAnchor="middle" fontSize="9"
          style={{ fill: 'var(--text-faint)', fontFamily: 'system-ui, sans-serif' }}>
          Safe
        </text>
        <text x={arcRight} y={cy + 18} textAnchor="middle" fontSize="9"
          style={{ fill: 'var(--text-faint)', fontFamily: 'system-ui, sans-serif' }}>
          Spam
        </text>
      </svg>
    </div>
  );
}
