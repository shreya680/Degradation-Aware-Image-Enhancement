import React from 'react';

/**
 * LoadingSpinner – animated spinner.
 * Props: text (optional label), size (px, default 32), color
 */
export default function LoadingSpinner({ text, size = 32, color = '#6366f1' }) {
  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: '0.75rem',
        padding: '1.5rem',
      }}
    >
      <svg
        width={size}
        height={size}
        viewBox="0 0 50 50"
        style={{ animation: 'daie-spin 0.8s linear infinite' }}
      >
        <style>{`@keyframes daie-spin { to { transform: rotate(360deg); } }`}</style>
        <circle
          cx="25"
          cy="25"
          r="20"
          fill="none"
          stroke={color}
          strokeWidth="4"
          strokeDasharray="90 60"
          strokeLinecap="round"
        />
      </svg>
      {text && (
        <span style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', fontWeight: 500 }}>
          {text}
        </span>
      )}
    </div>
  );
}
