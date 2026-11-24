import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Image, Download, ZoomIn, Columns, Sparkles } from 'lucide-react';
import ComparisonSlider from '../components/ComparisonSlider.jsx';
import { getSessionData } from '../services/storage.js';

export default function ComparisonPage() {
  const navigate  = useNavigate();
  const session   = getSessionData();

  const originalUrl = session.originalUrl  ?? null;
  const enhancedUrl = session.enhancedUrl  ?? null;

  const [zoom,    setZoom]    = useState(1);
  const [sideBySide, setSideBySide] = useState(false);

  if (!originalUrl || !enhancedUrl) {
    return (
      <div className="page-wrapper" style={{ display:'flex', alignItems:'center', justifyContent:'center', minHeight:'60vh' }}>
        <div className="card glass-panel" style={{ textAlign:'center', maxWidth:420, padding:'3rem 2rem' }}>
          <Image size={48} style={{ color:'var(--text-muted)', marginBottom:'1rem' }} />
          <h2 style={{ fontWeight:700, marginBottom:'.5rem' }}>No Images Loaded</h2>
          <p style={{ color:'var(--text-secondary)', marginBottom:'1.5rem' }}>
            Enhance an image first to use the comparison view.
          </p>
          <button className="btn btn-primary" onClick={() => navigate('/enhance')}>
            <Sparkles size={16} /> Go to Enhancement
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="page-wrapper">
      <div className="page-header" style={{ display:'flex', justifyContent:'space-between', alignItems:'flex-start', flexWrap:'wrap', gap:'1rem' }}>
        <div>
          <h1>Before &amp; After Comparison</h1>
          <p>Drag the slider to compare the original and enhanced images.</p>
        </div>
        <div style={{ display:'flex', gap:'.65rem', flexWrap:'wrap' }}>
          <button
            className={`btn ${sideBySide ? 'btn-primary' : 'btn-secondary'}`}
            onClick={() => setSideBySide(v => !v)}
          >
            <Columns size={15} /> {sideBySide ? 'Slider View' : 'Side-by-Side'}
          </button>
          <button
            className="btn btn-secondary"
            onClick={() => setZoom(z => z === 1 ? 2 : 1)}
          >
            <ZoomIn size={15} /> {zoom}× Zoom
          </button>
          <a href={enhancedUrl} download className="btn btn-primary">
            <Download size={15} /> Download Enhanced
          </a>
        </div>
      </div>

      <div className="card" style={{ overflow:'hidden' }}>
        {sideBySide ? (
          <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:'1rem' }}>
            <div>
              <p style={{ textAlign:'center', fontSize:'.85rem', color:'var(--text-secondary)', marginBottom:'.5rem', fontWeight:600 }}>Original</p>
              <div style={{ transform:`scale(${zoom})`, transformOrigin:'top left', transition:'transform .2s' }}>
                <img src={originalUrl} alt="Original" style={{ width:'100%', borderRadius:'var(--radius-md)', display:'block' }} />
              </div>
            </div>
            <div>
              <p style={{ textAlign:'center', fontSize:'.85rem', color:'var(--text-secondary)', marginBottom:'.5rem', fontWeight:600 }}>Enhanced</p>
              <div style={{ transform:`scale(${zoom})`, transformOrigin:'top left', transition:'transform .2s' }}>
                <img src={enhancedUrl} alt="Enhanced" style={{ width:'100%', borderRadius:'var(--radius-md)', display:'block' }} />
              </div>
            </div>
          </div>
        ) : (
          <div style={{ transform:`scale(${zoom})`, transformOrigin:'top left', transition:'transform .2s' }}>
            <ComparisonSlider originalSrc={originalUrl} enhancedSrc={enhancedUrl} height={520} />
          </div>
        )}

        <div style={{ marginTop:'1rem', display:'flex', gap:'1rem', justifyContent:'center', flexWrap:'wrap' }}>
          <div className="badge badge-blue">← Drag slider to compare →</div>
          <div className="badge badge-violet">Zoom: {zoom}×</div>
          {session.enhanceResult?.strategy && (
            <div className="badge badge-green">Strategy: {session.enhanceResult.strategy}</div>
          )}
        </div>
      </div>

      {/* Session info */}
      {session.analysis && (
        <div className="card" style={{ marginTop:'1.25rem' }}>
          <div className="section-title">Enhancement Details</div>
          <div style={{ display:'flex', gap:'1rem', flexWrap:'wrap' }}>
            <div style={{ flex:1, minWidth:160, background:'var(--bg-secondary)', borderRadius:'var(--radius-sm)', padding:'.75rem', fontSize:'.85rem' }}>
              <span style={{ color:'var(--text-muted)', display:'block', marginBottom:3 }}>Degradation Type</span>
              <span style={{ fontWeight:700, color:'var(--text-primary)' }}>{session.analysis.degradation_type ?? '—'}</span>
            </div>
            <div style={{ flex:1, minWidth:160, background:'var(--bg-secondary)', borderRadius:'var(--radius-sm)', padding:'.75rem', fontSize:'.85rem' }}>
              <span style={{ color:'var(--text-muted)', display:'block', marginBottom:3 }}>Severity</span>
              <span style={{ fontWeight:700, color:'var(--text-primary)' }}>{session.analysis.severity_level ?? '—'}</span>
            </div>
            {session.enhanceResult?.processing_time_ms != null && (
              <div style={{ flex:1, minWidth:160, background:'var(--bg-secondary)', borderRadius:'var(--radius-sm)', padding:'.75rem', fontSize:'.85rem' }}>
                <span style={{ color:'var(--text-muted)', display:'block', marginBottom:3 }}>Processing Time</span>
                <span style={{ fontWeight:700, color:'var(--text-primary)' }}>{session.enhanceResult.processing_time_ms} ms</span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
