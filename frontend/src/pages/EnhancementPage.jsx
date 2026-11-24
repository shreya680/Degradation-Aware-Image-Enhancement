import React, { useState, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Upload, Image, Sparkles, AlertCircle, CheckCircle,
  Download, BarChart3, ScanSearch, Trash2, RefreshCw,
  ArrowRight, Info, ChevronDown,
} from 'lucide-react';
import LoadingSpinner from '../components/ui/LoadingSpinner.jsx';
import StatusBadge    from '../components/ui/StatusBadge.jsx';
import ImagePreview   from '../components/ui/ImagePreview.jsx';
import { uploadImage, analyzeImage, enhanceImage } from '../services/api.js';
import { saveSessionData } from '../services/storage.js';

const MAX_SIZE = 10 * 1024 * 1024; // 10 MB

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function ScoreBar({ label, value }) {
  const pct = Math.round((value ?? 0) * 100);
  const color = pct > 60 ? '#ef4444' : pct > 35 ? '#f59e0b' : '#10b981';
  return (
    <div style={{ marginBottom: '.45rem' }}>
      <div style={{ display:'flex', justifyContent:'space-between', fontSize:'.78rem', marginBottom:2 }}>
        <span style={{ color:'var(--text-secondary)', textTransform:'capitalize' }}>{label}</span>
        <span style={{ color:'var(--text-primary)', fontWeight:600 }}>{pct}%</span>
      </div>
      <div style={{ height:5, background:'var(--bg-secondary)', borderRadius:3, overflow:'hidden' }}>
        <div style={{ width:`${pct}%`, height:'100%', background:color, borderRadius:3, transition:'width .4s' }} />
      </div>
    </div>
  );
}

export default function EnhancementPage() {
  const navigate = useNavigate();
  const fileInputRef = useRef(null);

  // ── State ──
  const [file,          setFile]          = useState(null);
  const [preview,       setPreview]       = useState(null);  // local object URL
  const [fileMeta,      setFileMeta]      = useState(null);  // { name, size, dimensions }
  const [uploadedName,  setUploadedName]  = useState(null);  // server filename
  const [dragging,      setDragging]      = useState(false);

  const [analyzing,     setAnalyzing]     = useState(false);
  const [analysis,      setAnalysis]      = useState(null);
  const [analyzeError,  setAnalyzeError]  = useState(null);

  const [enhancing,     setEnhancing]     = useState(false);
  const [enhanced,      setEnhanced]      = useState(null);  // { url, filename, dimensions }
  const [enhanceError,  setEnhanceError]  = useState(null);

  const [uploadLoading, setUploadLoading] = useState(false);
  const [uploadError,   setUploadError]   = useState(null);

  // ── File selection / upload ──
  const handleFile = useCallback(async (f) => {
    if (!f) return;
    if (!['image/jpeg', 'image/jpg', 'image/png'].includes(f.type)) {
      setUploadError('Only JPEG and PNG files are supported.');
      return;
    }
    if (f.size > MAX_SIZE) {
      setUploadError(`File too large. Max 10 MB (got ${formatBytes(f.size)}).`);
      return;
    }
    setUploadError(null);
    setAnalysis(null);
    setEnhanced(null);
    setEnhanceError(null);
    setAnalyzeError(null);

    // local preview
    const url = URL.createObjectURL(f);
    setPreview(url);
    setFile(f);

    // read dimensions
    const img = new window.Image();
    img.onload = () => {
      setFileMeta({ name: f.name, size: formatBytes(f.size), dimensions: `${img.naturalWidth} × ${img.naturalHeight}` });
    };
    img.src = url;

    // upload to server
    setUploadLoading(true);
    try {
      const res = await uploadImage(f);
      setUploadedName(res.data.filename);
      setFileMeta(prev => ({ ...prev, dimensions: res.data.dimensions ?? prev?.dimensions }));
      // auto-analyze
      await runAnalyze(res.data.filename);
    } catch (e) {
      setUploadError(e.response?.data?.detail ?? e.message);
    } finally {
      setUploadLoading(false);
    }
  }, []);

  const runAnalyze = async (fname) => {
    setAnalyzing(true);
    setAnalyzeError(null);
    try {
      const res = await analyzeImage(fname);
      setAnalysis(res.data);
      saveSessionData({ originalFilename: fname, analysis: res.data });
    } catch (e) {
      setAnalyzeError(e.response?.data?.detail ?? e.message);
    } finally {
      setAnalyzing(false);
    }
  };

  const runEnhance = async () => {
    if (!uploadedName) return;
    setEnhancing(true);
    setEnhanceError(null);
    try {
      const res = await enhanceImage(uploadedName);
      const enhData = {
        url: `/images/${res.data.enhanced_filename}`,
        filename: res.data.enhanced_filename,
        dimensions: res.data.dimensions,
        strategy: res.data.strategy,
      };
      setEnhanced(enhData);
      saveSessionData({
        originalUrl: preview,
        enhancedUrl: enhData.url,
        enhancedFilename: enhData.filename,
        originalFilename: uploadedName,
        enhanceResult: res.data,
      });
    } catch (e) {
      setEnhanceError(e.response?.data?.detail ?? e.message);
    } finally {
      setEnhancing(false);
    }
  };

  const handleReset = () => {
    setFile(null); setPreview(null); setFileMeta(null); setUploadedName(null);
    setAnalysis(null); setEnhanced(null);
    setUploadError(null); setAnalyzeError(null); setEnhanceError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  // ── Drag & drop ──
  const onDragOver  = (e) => { e.preventDefault(); setDragging(true);  };
  const onDragLeave = ()  => setDragging(false);
  const onDrop      = (e) => { e.preventDefault(); setDragging(false); handleFile(e.dataTransfer.files[0]); };

  const severityStatus = analysis?.severity_level?.toLowerCase() === 'high' ? 'error'
    : analysis?.severity_level?.toLowerCase() === 'medium' ? 'loading' : 'loaded';

  return (
    <div className="page-wrapper">
      <div className="page-header">
        <h1>Image Enhancement Workspace</h1>
        <p>Upload a degraded image to analyze and enhance it using the DAIE pipeline.</p>
      </div>

      <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:'1.5rem' }}>

        {/* ── LEFT COLUMN ── */}
        <div style={{ display:'flex', flexDirection:'column', gap:'1.25rem' }}>

          {/* Upload Zone */}
          <div className="card">
            <div className="section-title"><Upload size={18} /> Upload Image</div>

            <div
              onDragOver={onDragOver}
              onDragLeave={onDragLeave}
              onDrop={onDrop}
              onClick={() => !uploadLoading && fileInputRef.current?.click()}
              style={{
                border: `2px dashed ${dragging ? 'var(--accent-blue)' : 'var(--border-color)'}`,
                borderRadius: 'var(--radius-md)',
                padding: '2.5rem 1.5rem',
                textAlign: 'center',
                cursor: uploadLoading ? 'default' : 'pointer',
                transition: 'border-color .2s, background .2s',
                background: dragging ? 'rgba(59,130,246,0.06)' : 'transparent',
              }}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".jpg,.jpeg,.png"
                style={{ display:'none' }}
                onChange={e => handleFile(e.target.files[0])}
              />
              {uploadLoading ? (
                <LoadingSpinner text="Uploading…" />
              ) : (
                <>
                  <Image size={40} style={{ color:'var(--text-muted)', marginBottom:'.75rem' }} />
                  <p style={{ fontWeight:600, marginBottom:'.35rem' }}>Drag &amp; drop an image here</p>
                  <p style={{ color:'var(--text-secondary)', fontSize:'.85rem', marginBottom:'1rem' }}>or click to browse files</p>
                  <span className="btn btn-secondary" style={{ pointerEvents:'none' }}>Browse Files</span>
                  <p style={{ color:'var(--text-muted)', fontSize:'.75rem', marginTop:'.85rem' }}>
                    Accepted: JPEG, PNG &nbsp;·&nbsp; Max 10 MB
                  </p>
                </>
              )}
            </div>

            {uploadError && (
              <div className="alert alert-error" style={{ marginTop:'.75rem' }}>
                <AlertCircle size={15} /> {uploadError}
              </div>
            )}

            {fileMeta && (
              <div style={{ marginTop:'1rem', background:'var(--bg-secondary)', borderRadius:'var(--radius-sm)', padding:'.75rem 1rem', fontSize:'.82rem' }}>
                <div style={{ display:'flex', justifyContent:'space-between', flexWrap:'wrap', gap:'.4rem' }}>
                  <span style={{ color:'var(--text-primary)', fontWeight:600, fontFamily:'monospace' }}>{fileMeta.name}</span>
                  <div style={{ display:'flex', gap:'1rem', color:'var(--text-muted)' }}>
                    <span>{fileMeta.dimensions}</span>
                    <span>{fileMeta.size}</span>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Original Preview */}
          {preview && (
            <div className="card">
              <div className="section-title"><Image size={18} /> Original Image</div>
              <ImagePreview src={preview} label="Original" dimensions={fileMeta?.dimensions} fileSize={fileMeta?.size} />
            </div>
          )}

          {/* Degradation Analysis Panel */}
          {(analyzing || analysis || analyzeError) && (
            <div className="card">
              <div className="section-title"><Info size={18} /> Degradation Analysis</div>
              {analyzing && <LoadingSpinner text="Analyzing image…" />}
              {analyzeError && (
                <div className="alert alert-error"><AlertCircle size={15} /> {analyzeError}</div>
              )}
              {analysis && !analyzing && (
                <div style={{ display:'flex', flexDirection:'column', gap:'.85rem' }}>
                  <div style={{ display:'flex', gap:'.65rem', flexWrap:'wrap' }}>
                    <span className="badge badge-blue">{analysis.degradation_type}</span>
                    <StatusBadge status={severityStatus} label={`${analysis.severity_level} Severity`} />
                    {analysis.confidence != null && (
                      <span className="badge badge-violet">{Math.round(analysis.confidence * 100)}% confidence</span>
                    )}
                  </div>

                  {analysis.all_scores && (
                    <div>
                      <p style={{ fontSize:'.78rem', color:'var(--text-secondary)', marginBottom:'.5rem', fontWeight:600 }}>SCORE BREAKDOWN</p>
                      {Object.entries(analysis.all_scores).map(([k, v]) => (
                        <ScoreBar key={k} label={k.replace(/_/g,' ')} value={v} />
                      ))}
                    </div>
                  )}

                  {analysis.recommended_strategy && (
                    <div style={{ background:'var(--bg-secondary)', borderRadius:'var(--radius-sm)', padding:'.65rem .9rem', fontSize:'.82rem' }}>
                      <span style={{ color:'var(--text-muted)', fontWeight:600, marginRight:6 }}>STRATEGY:</span>
                      <span style={{ color:'var(--text-primary)' }}>{analysis.recommended_strategy}</span>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>

        {/* ── RIGHT COLUMN ── */}
        <div style={{ display:'flex', flexDirection:'column', gap:'1.25rem' }}>

          {/* Enhance Button */}
          <div className="card" style={{ display:'flex', flexDirection:'column', gap:'1rem' }}>
            <div className="section-title"><Sparkles size={18} /> Enhance Image</div>
            <p style={{ color:'var(--text-secondary)', fontSize:'.875rem' }}>
              Run the adaptive enhancement pipeline. The strategy is chosen based on the detected degradation type.
            </p>
            <button
              className="btn btn-primary"
              onClick={runEnhance}
              disabled={!uploadedName || enhancing || uploadLoading}
              style={{ width:'100%', justifyContent:'center', padding:'.85rem' }}
            >
              {enhancing ? <><RefreshCw size={16} style={{ animation:'daie-spin .8s linear infinite' }} /> Enhancing…</> : <><Sparkles size={16} /> Enhance Image</>}
            </button>
            {!uploadedName && (
              <p style={{ color:'var(--text-muted)', fontSize:'.8rem', textAlign:'center' }}>↑ Upload an image first</p>
            )}
            {enhanceError && (
              <div className="alert alert-error"><AlertCircle size={15} /> {enhanceError}</div>
            )}
          </div>

          {/* Enhanced Preview */}
          {enhanced && (
            <div className="card">
              <div className="section-title"><CheckCircle size={18} style={{color:'#10b981'}} /> Enhanced Image</div>
              <ImagePreview src={enhanced.url} label="Enhanced" dimensions={enhanced.dimensions} />

              {enhanced.strategy && (
                <div style={{ marginTop:'.85rem', background:'var(--bg-secondary)', borderRadius:'var(--radius-sm)', padding:'.65rem .9rem', fontSize:'.82rem' }}>
                  <span style={{ color:'var(--text-muted)', fontWeight:600, marginRight:6 }}>Applied strategy:</span>
                  <span style={{ color:'#10b981' }}>{enhanced.strategy}</span>
                </div>
              )}

              {/* Action buttons */}
              <div style={{ display:'flex', gap:'.65rem', marginTop:'1rem', flexWrap:'wrap' }}>
                <button className="btn btn-secondary" onClick={() => navigate('/compare')} style={{ flex:1, justifyContent:'center' }}>
                  <Image size={15} /> Compare <ArrowRight size={13} />
                </button>
                <button className="btn btn-secondary" onClick={() => navigate('/detect')} style={{ flex:1, justifyContent:'center' }}>
                  <ScanSearch size={15} /> Run Detection
                </button>
                <button className="btn btn-secondary" onClick={() => navigate('/analytics')} style={{ flex:1, justifyContent:'center' }}>
                  <BarChart3 size={15} /> Analytics
                </button>
              </div>

              <div style={{ display:'flex', gap:'.65rem', marginTop:'.65rem' }}>
                <a
                  href={enhanced.url}
                  download
                  className="btn btn-primary"
                  style={{ flex:1, justifyContent:'center' }}
                >
                  <Download size={15} /> Download Enhanced
                </a>
                <button className="btn btn-danger" onClick={handleReset} style={{ flex:'0 0 auto' }}>
                  <Trash2 size={15} /> Reset
                </button>
              </div>
            </div>
          )}

          {/* Instructions if nothing uploaded yet */}
          {!preview && (
            <div className="card glass-panel" style={{ textAlign:'center', padding:'2.5rem 1.5rem' }}>
              <Sparkles size={40} style={{ color:'var(--accent-violet)', marginBottom:'1rem', opacity:.7 }} />
              <h3 style={{ fontWeight:700, marginBottom:'.5rem' }}>How it works</h3>
              <ol style={{ color:'var(--text-secondary)', fontSize:'.875rem', textAlign:'left', paddingLeft:'1.25rem', lineHeight:2 }}>
                <li>Upload a JPEG or PNG image (max 10 MB)</li>
                <li>Automatic degradation analysis runs immediately</li>
                <li>Click <strong style={{color:'var(--text-primary)'}}>Enhance Image</strong> to run the AI pipeline</li>
                <li>Compare, detect objects, and review analytics</li>
              </ol>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
