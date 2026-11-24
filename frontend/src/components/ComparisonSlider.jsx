import React, { useState, useRef, useEffect, useCallback } from 'react';
import { MoveHorizontal } from 'lucide-react';

/**
 * ComparisonSlider – interactive before/after image comparison.
 * Props: originalSrc, enhancedSrc, height (px)
 */
export default function ComparisonSlider({ originalSrc, enhancedSrc, height = 480 }) {
  const [position, setPosition] = useState(50); // 0-100 percentage
  const containerRef = useRef(null);
  const isDragging   = useRef(false);

  const updatePosition = useCallback((clientX) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x    = Math.max(0, Math.min(clientX - rect.left, rect.width));
    setPosition((x / rect.width) * 100);
  }, []);

  // Mouse events
  const onMouseDown = (e) => {
    isDragging.current = true;
    updatePosition(e.clientX);
    e.preventDefault();
  };
  const onMouseMove = useCallback((e) => {
    if (!isDragging.current) return;
    updatePosition(e.clientX);
  }, [updatePosition]);
  const onMouseUp = useCallback(() => { isDragging.current = false; }, []);

  // Touch events
  const onTouchStart = (e) => {
    isDragging.current = true;
    updatePosition(e.touches[0].clientX);
  };
  const onTouchMove = useCallback((e) => {
    if (!isDragging.current) return;
    updatePosition(e.touches[0].clientX);
    e.preventDefault();
  }, [updatePosition]);
  const onTouchEnd = useCallback(() => { isDragging.current = false; }, []);

  useEffect(() => {
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup',   onMouseUp);
    window.addEventListener('touchmove', onTouchMove, { passive: false });
    window.addEventListener('touchend',  onTouchEnd);
    return () => {
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup',   onMouseUp);
      window.removeEventListener('touchmove', onTouchMove);
      window.removeEventListener('touchend',  onTouchEnd);
    };
  }, [onMouseMove, onMouseUp, onTouchMove, onTouchEnd]);

  const sliderStyle = {
    position: 'relative',
    width: '100%',
    height,
    overflow: 'hidden',
    borderRadius: 'var(--radius-lg)',
    border: '1px solid var(--border-color)',
    cursor: 'col-resize',
    userSelect: 'none',
    background: '#000',
  };

  const imgBase = {
    position: 'absolute',
    top: 0,
    left: 0,
    width: '100%',
    height: '100%',
    objectFit: 'contain',
    pointerEvents: 'none',
  };

  return (
    <div style={sliderStyle} ref={containerRef} onMouseDown={onMouseDown} onTouchStart={onTouchStart}>
      {/* Enhanced (right / background) */}
      <img src={enhancedSrc} alt="Enhanced" style={imgBase} />

      {/* Original clipped to left */}
      <div
        style={{
          position: 'absolute',
          top: 0, left: 0,
          width: `${position}%`,
          height: '100%',
          overflow: 'hidden',
        }}
      >
        <img
          src={originalSrc}
          alt="Original"
          style={{ ...imgBase, width: `${10000 / position}%`, maxWidth: 'none' }}
        />
      </div>

      {/* Divider line */}
      <div
        style={{
          position: 'absolute',
          top: 0,
          left: `${position}%`,
          transform: 'translateX(-50%)',
          width: 3,
          height: '100%',
          background: 'linear-gradient(to bottom, #3b82f6, #8b5cf6)',
          pointerEvents: 'none',
        }}
      />

      {/* Handle */}
      <div
        style={{
          position: 'absolute',
          top: '50%',
          left: `${position}%`,
          transform: 'translate(-50%, -50%)',
          width: 36,
          height: 36,
          borderRadius: '50%',
          background: 'linear-gradient(135deg, #3b82f6, #8b5cf6)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          boxShadow: '0 2px 12px rgba(0,0,0,0.5)',
          pointerEvents: 'none',
          border: '2px solid rgba(255,255,255,0.3)',
        }}
      >
        <MoveHorizontal size={16} color="#fff" />
      </div>

      {/* Labels */}
      <div style={{ position:'absolute', top:12, left:12, padding:'2px 10px', background:'rgba(0,0,0,0.6)', borderRadius:6, fontSize:'0.75rem', color:'#f1f5f9', pointerEvents:'none' }}>
        Original
      </div>
      <div style={{ position:'absolute', top:12, right:12, padding:'2px 10px', background:'rgba(0,0,0,0.6)', borderRadius:6, fontSize:'0.75rem', color:'#f1f5f9', pointerEvents:'none' }}>
        Enhanced
      </div>
    </div>
  );
}
