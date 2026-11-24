import React from 'react';

/**
 * MetricCard – displays a single KPI tile.
 * Props: title, value, unit, icon (JSX), color (css var or hex), description
 */
export default function MetricCard({ title, value, unit = '', icon, color = '#3b82f6', description }) {
  return (
    <div
      className="card"
      style={{
        borderLeft: `3px solid ${color}`,
        display: 'flex',
        flexDirection: 'column',
        gap: '0.5rem',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <span style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', fontWeight: 600, letterSpacing: '0.03em', textTransform: 'uppercase' }}>
          {title}
        </span>
        {icon && (
          <span style={{ color, opacity: 0.85 }}>
            {icon}
          </span>
        )}
      </div>
      <div style={{ fontSize: '1.9rem', fontWeight: 800, color: 'var(--text-primary)', lineHeight: 1 }}>
        {value ?? <span style={{ color: 'var(--text-muted)', fontSize:'1.2rem' }}>—</span>}
        {value != null && unit && (
          <span style={{ fontSize: '0.95rem', fontWeight: 500, color: 'var(--text-secondary)', marginLeft: 4 }}>
            {unit}
          </span>
        )}
      </div>
      {description && (
        <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', margin: 0 }}>{description}</p>
      )}
    </div>
  );
}
