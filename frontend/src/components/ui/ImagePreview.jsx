import React from 'react';
import { ImageIcon } from 'lucide-react';

/**
 * ImagePreview – shows an image with metadata footer.
 * Props: src, label, dimensions, fileSize, style
 */
export default function ImagePreview({ src, label, dimensions, fileSize, style = {} }) {
  if (!src) {
    return (
      <div
        style={{
          background: 'var(--bg-secondary)',
          border: '2px dashed var(--border-color)',
          borderRadius: 'var(--radius-md)',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '0.5rem',
          minHeight: 200,
          color: 'var(--text-muted)',
          ...style,
        }}
      >
        <ImageIcon size={32} opacity={0.4} />
        <span style={{ fontSize: '0.85rem' }}>No image</span>
      </div>
    );
  }

  return (
    <div
      style={{
        borderRadius: 'var(--radius-md)',
        overflow: 'hidden',
        border: '1px solid var(--border-color)',
        background: '#000',
        ...style,
      }}
    >
      <img
        src={src}
        alt={label || 'Image'}
        style={{ width: '100%', display: 'block', maxHeight: 480, objectFit: 'contain' }}
      />
      <div
        style={{
          background: 'var(--bg-card)',
          padding: '0.6rem 1rem',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '0.4rem',
        }}
      >
        <span style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-primary)' }}>{label}</span>
        <div style={{ display: 'flex', gap: '0.75rem', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          {dimensions && <span>{dimensions}</span>}
          {fileSize  && <span>{fileSize}</span>}
        </div>
      </div>
    </div>
  );
}
