import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  ScanSearch,
  Sparkles,
  Sliders,
  CheckCircle,
  AlertCircle,
  Download,
  BarChart2,
  Image as ImageIcon,
  ArrowRight,
  TrendingUp,
  ShieldCheck,
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import LoadingSpinner from '../components/ui/LoadingSpinner.jsx';
import { detectObjects } from '../services/api.js';
import { getSessionData, saveSessionData } from '../services/storage.js';

export default function DetectionPage() {
  const navigate = useNavigate();
  const session = getSessionData();

  const [origFilename, setOrigFilename] = useState(session.originalFilename || '');
  const [enhFilename, setEnhFilename] = useState(session.enhancedFilename || '');
  const [confThreshold, setConfThreshold] = useState(0.25);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [results, setResults] = useState(session.detectionResults || null);

  useEffect(() => {
    // If not already detected but filenames exist, allow user to trigger
    if (session.originalFilename) setOrigFilename(session.originalFilename);
    if (session.enhancedFilename) setEnhFilename(session.enhancedFilename);
  }, []);

  const handleRunDetection = async () => {
    if (!origFilename) {
      setError('Please enhance or upload an image first.');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const res = await detectObjects(origFilename, enhFilename || null, confThreshold);
      setResults(res.data);
      saveSessionData({
        detectionResults: res.data,
      });
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Detection failed');
    } finally {
      setLoading(false);
    }
  };

  // Prepare chart data: class name breakdown
  const chartData = React.useMemo(() => {
    if (!results) return [];
    const counts = {};
    (results.original?.detections || []).forEach((d) => {
      counts[d.class_name] = counts[d.class_name] || { class: d.class_name, original: 0, enhanced: 0 };
      counts[d.class_name].original += 1;
    });
    (results.enhanced?.detections || []).forEach((d) => {
      counts[d.class_name] = counts[d.class_name] || { class: d.class_name, original: 0, enhanced: 0 };
      counts[d.class_name].enhanced += 1;
    });
    return Object.values(counts);
  }, [results]);

  const origCount = results?.original?.count ?? 0;
  const enhCount = results?.enhanced?.count ?? 0;
  const diff = enhCount - origCount;

  return (
    <div className="page-wrapper" style={{ padding: '2rem', maxWidth: 1400, margin: '0 auto' }}>
      {/* Header */}
      <div className="page-header" style={{ marginBottom: '2rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
          <div style={{
            background: 'linear-gradient(135deg, #3b82f6, #6366f1)',
            padding: '0.6rem',
            borderRadius: '12px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}>
            <ScanSearch size={24} color="#fff" />
          </div>
          <div>
            <h1 style={{ fontSize: '1.75rem', fontWeight: 700, margin: 0 }}>Downstream Object Detection (YOLOv8)</h1>
            <p style={{ color: 'var(--text-secondary)', margin: 0, fontSize: '0.9rem' }}>
              Evaluate downstream task preservation: comparing detections on raw degraded vs. enhanced images.
            </p>
          </div>
        </div>
      </div>

      {/* Control Bar */}
      <div className="card" style={{ marginBottom: '2rem', display: 'flex', flexWrap: 'wrap', gap: '1.5rem', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '2rem', flex: 1, minWidth: 300 }}>
          <div style={{ flex: 1, maxWidth: 360 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.4rem', fontSize: '0.85rem' }}>
              <span style={{ fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <Sliders size={14} /> Confidence Threshold:
              </span>
              <span style={{ color: 'var(--accent-blue)', fontWeight: 700 }}>{(confThreshold * 100).toFixed(0)}%</span>
            </div>
            <input
              type="range"
              min="0.05"
              max="0.95"
              step="0.05"
              value={confThreshold}
              onChange={(e) => setConfThreshold(parseFloat(e.target.value))}
              style={{ width: '100%', accentColor: 'var(--accent-blue)', cursor: 'pointer' }}
            />
          </div>

          <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
            <div>Target: <code style={{ color: 'var(--text-primary)' }}>{origFilename || 'No file selected'}</code></div>
            <div>Enhanced: <code style={{ color: 'var(--text-primary)' }}>{enhFilename || 'None'}</code></div>
          </div>
        </div>

        <div style={{ display: 'flex', gap: '0.75rem' }}>
          {!origFilename && (
            <button className="btn btn-secondary" onClick={() => navigate('/enhance')}>
              <Sparkles size={16} /> Load Image
            </button>
          )}
          <button
            className="btn btn-primary"
            onClick={handleRunDetection}
            disabled={loading || !origFilename}
            style={{ minWidth: 160 }}
          >
            {loading ? <LoadingSpinner text="Running YOLOv8..." /> : (
              <>
                <ScanSearch size={16} /> Run Detection
              </>
            )}
          </button>
        </div>
      </div>

      {error && (
        <div className="card" style={{ marginBottom: '1.5rem', borderColor: 'var(--accent-red)', background: 'rgba(239, 68, 68, 0.1)', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <AlertCircle size={20} color="var(--accent-red)" />
          <span style={{ color: 'var(--accent-red)', fontSize: '0.9rem' }}>{error}</span>
        </div>
      )}

      {/* Metrics Banner */}
      {results && (
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
          gap: '1rem',
          marginBottom: '2rem',
        }}>
          <div className="card" style={{ borderLeft: '4px solid var(--accent-yellow)' }}>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Original Detections</div>
            <div style={{ fontSize: '1.8rem', fontWeight: 800, color: 'var(--text-primary)' }}>{origCount}</div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Low-light / degraded image</div>
          </div>

          <div className="card" style={{ borderLeft: '4px solid var(--accent-green)' }}>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Enhanced Detections</div>
            <div style={{ fontSize: '1.8rem', fontWeight: 800, color: 'var(--accent-green)' }}>{enhCount}</div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Adaptive Zero-DCE enhanced</div>
          </div>

          <div className="card" style={{ borderLeft: `4px solid ${diff >= 0 ? 'var(--accent-blue)' : 'var(--accent-red)'}` }}>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Downstream Delta</div>
            <div style={{
              fontSize: '1.8rem',
              fontWeight: 800,
              color: diff >= 0 ? 'var(--accent-blue)' : 'var(--accent-red)',
              display: 'flex',
              alignItems: 'center',
              gap: '0.4rem',
            }}>
              {diff > 0 ? `+${diff}` : diff}
              <TrendingUp size={20} />
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              {diff > 0 ? 'Improved object discoverability' : diff === 0 ? 'Maintained object stability' : 'False-positive suppression'}
            </div>
          </div>

          <div className="card" style={{ borderLeft: '4px solid var(--accent-purple)' }}>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Model Used</div>
            <div style={{ fontSize: '1.3rem', fontWeight: 700, color: 'var(--text-primary)', marginTop: '0.3rem' }}>YOLOv8 Nano</div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Confidence ≥ {(confThreshold * 100).toFixed(0)}%</div>
          </div>
        </div>
      )}

      {/* Visual Bounding Boxes Comparison */}
      {results && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(450px, 1fr))', gap: '1.5rem', marginBottom: '2rem' }}>
          {/* Original with Annotations */}
          <div className="card" style={{ display: 'flex', flexDirection: 'column' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <div style={{ fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <ImageIcon size={18} color="var(--accent-yellow)" />
                Original Image Detections ({origCount})
              </div>
              {results.original?.annotated_url && (
                <a href={results.original.annotated_url} download className="btn btn-secondary" style={{ padding: '0.35rem 0.75rem', fontSize: '0.75rem' }}>
                  <Download size={14} /> Download
                </a>
              )}
            </div>
            <div style={{
              background: '#070a12',
              borderRadius: '8px',
              overflow: 'hidden',
              minHeight: 320,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}>
              {results.original?.annotated_url ? (
                <img
                  src={results.original.annotated_url}
                  alt="Original Annotated"
                  style={{ width: '100%', maxHeight: 480, objectFit: 'contain' }}
                />
              ) : (
                <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>No detections to display</div>
              )}
            </div>
          </div>

          {/* Enhanced with Annotations */}
          <div className="card" style={{ display: 'flex', flexDirection: 'column' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <div style={{ fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Sparkles size={18} color="var(--accent-green)" />
                Enhanced Image Detections ({enhCount})
              </div>
              {results.enhanced?.annotated_url && (
                <a href={results.enhanced.annotated_url} download className="btn btn-secondary" style={{ padding: '0.35rem 0.75rem', fontSize: '0.75rem' }}>
                  <Download size={14} /> Download
                </a>
              )}
            </div>
            <div style={{
              background: '#070a12',
              borderRadius: '8px',
              overflow: 'hidden',
              minHeight: 320,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}>
              {results.enhanced?.annotated_url ? (
                <img
                  src={results.enhanced.annotated_url}
                  alt="Enhanced Annotated"
                  style={{ width: '100%', maxHeight: 480, objectFit: 'contain' }}
                />
              ) : (
                <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>No enhanced image provided or 0 detections</div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Class Comparison Chart and Table */}
      {results && chartData.length > 0 && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', marginBottom: '2rem' }}>
          {/* Chart */}
          <div className="card">
            <h3 style={{ fontSize: '1rem', fontWeight: 600, marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <BarChart2 size={18} color="var(--accent-blue)" /> Class Breakdown Comparison
            </h3>
            <div style={{ height: 260 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                  <XAxis dataKey="class" stroke="var(--text-muted)" fontSize={11} interval={0} angle={-25} textAnchor="end" />
                  <YAxis stroke="var(--text-muted)" fontSize={11} allowDecimals={false} />
                  <Tooltip
                    contentStyle={{ background: 'var(--bg-card)', borderColor: 'var(--border-color)', borderRadius: '8px', fontSize: '0.8rem' }}
                  />
                  <Legend wrapperStyle={{ fontSize: '0.8rem', paddingTop: '8px' }} />
                  <Bar dataKey="original" name="Original Image" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="enhanced" name="Enhanced Image" fill="#10b981" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Detections Detail Table */}
          <div className="card" style={{ overflowY: 'auto', maxHeight: 350 }}>
            <h3 style={{ fontSize: '1rem', fontWeight: 600, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <ShieldCheck size={18} color="var(--accent-purple)" /> Detection Confidence Log
            </h3>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.82rem' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border-color)', color: 'var(--text-secondary)', textAlign: 'left' }}>
                  <th style={{ padding: '0.5rem' }}>Source</th>
                  <th style={{ padding: '0.5rem' }}>Class</th>
                  <th style={{ padding: '0.5rem' }}>Confidence</th>
                  <th style={{ padding: '0.5rem' }}>Bounding Box</th>
                </tr>
              </thead>
              <tbody>
                {(results.enhanced?.detections || []).map((d, i) => (
                  <tr key={`enh-${i}`} style={{ borderBottom: '1px solid rgba(255,255,255,0.03)' }}>
                    <td style={{ padding: '0.45rem', color: 'var(--accent-green)', fontWeight: 600 }}>Enhanced</td>
                    <td style={{ padding: '0.45rem', fontWeight: 500 }}>{d.class_name}</td>
                    <td style={{ padding: '0.45rem' }}>{(d.confidence * 100).toFixed(1)}%</td>
                    <td style={{ padding: '0.45rem', color: 'var(--text-muted)', fontFamily: 'monospace' }}>
                      [{d.bbox.map((v) => Math.round(v)).join(', ')}]
                    </td>
                  </tr>
                ))}
                {(results.original?.detections || []).map((d, i) => (
                  <tr key={`orig-${i}`} style={{ borderBottom: '1px solid rgba(255,255,255,0.03)' }}>
                    <td style={{ padding: '0.45rem', color: 'var(--accent-yellow)', fontWeight: 600 }}>Original</td>
                    <td style={{ padding: '0.45rem', fontWeight: 500 }}>{d.class_name}</td>
                    <td style={{ padding: '0.45rem' }}>{(d.confidence * 100).toFixed(1)}%</td>
                    <td style={{ padding: '0.45rem', color: 'var(--text-muted)', fontFamily: 'monospace' }}>
                      [{d.bbox.map((v) => Math.round(v)).join(', ')}]
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Navigation Footer */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '2rem' }}>
        <button className="btn btn-secondary" onClick={() => navigate('/compare')}>
          Back to Comparison
        </button>
        <button className="btn btn-primary" onClick={() => navigate('/analytics')}>
          View Full Analytics <ArrowRight size={16} />
        </button>
      </div>
    </div>
  );
}
