import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Sparkles, Cpu, Image, Clock, CheckCircle, XCircle, Loader, ArrowRight, Eye, BarChart3, Database, Shield } from 'lucide-react';
import MetricCard   from '../components/ui/MetricCard.jsx';
import StatusBadge  from '../components/ui/StatusBadge.jsx';
import LoadingSpinner from '../components/ui/LoadingSpinner.jsx';
import { getHealth, getModelsStatus, getHistory } from '../services/api.js';

const capabilities = [
  { icon: Sparkles, color:'#6366f1', title:'Zero-DCE Enhancement',  desc:'Low-reference deep curve estimation for real-world low-light images.' },
  { icon: Eye,      color:'#3b82f6', title:'YOLOv8 Object Detection', desc:'State-of-the-art detection on both original and enhanced images.' },
  { icon: BarChart3,color:'#10b981', title:'Perceptual Metrics',     desc:'PSNR, SSIM, LPIPS and brightness analysis for objective evaluation.' },
  { icon: Database, color:'#f59e0b', title:'Benchmark Datasets',     desc:'Validated on LOL and ExDark datasets for reproducible results.' },
];

export default function Dashboard() {
  const navigate = useNavigate();
  const [health,  setHealth]  = useState(null);
  const [models,  setModels]  = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState(null);

  useEffect(() => {
    let alive = true;
    const fetchAll = async () => {
      setLoading(true);
      try {
        const [h, m, hist] = await Promise.allSettled([getHealth(), getModelsStatus(), getHistory()]);
        if (!alive) return;
        if (h.status      === 'fulfilled') setHealth(h.value.data);
        if (m.status      === 'fulfilled') setModels(m.value.data);
        if (hist.status   === 'fulfilled') setHistory(hist.value.data?.history ?? hist.value.data ?? []);
      } catch (e) {
        if (alive) setError(e.message);
      } finally {
        if (alive) setLoading(false);
      }
    };
    fetchAll();
    return () => { alive = false; };
  }, []);

  const zdStatus  = models?.zero_dce?.loaded ? 'loaded' : models ? 'error' : 'loading';
  const yoloStatus = models?.yolov8?.loaded  ? 'loaded' : models ? 'error' : 'loading';
  const device    = health?.device ?? '—';
  const processed = history.length;

  return (
    <div className="page-wrapper">
      {/* Header */}
      <div className="page-header" style={{ display:'flex', justifyContent:'space-between', alignItems:'flex-start', flexWrap:'wrap', gap:'1rem' }}>
        <div>
          <h1>Dashboard</h1>
          <p>Degradation-Aware Image Enhancement System — overview &amp; status.</p>
        </div>
        <button className="btn btn-primary" onClick={() => navigate('/enhance')}>
          <Sparkles size={16} /> Enhance an Image <ArrowRight size={14} />
        </button>
      </div>

      {error && (
        <div className="alert alert-warn" style={{ marginBottom:'1.5rem' }}>
          <Shield size={16} /> Backend unreachable: {error}. Start the FastAPI server on port 8000.
        </div>
      )}

      {/* Stat Cards */}
      <div className="grid-4" style={{ marginBottom:'1.5rem' }}>
        <MetricCard
          title="ZeroDCE Status"
          value={<StatusBadge status={zdStatus} label={zdStatus === 'loaded' ? 'Loaded' : zdStatus === 'error' ? 'Not Loaded' : 'Loading…'} />}
          icon={<Sparkles size={18} />}
          color="#6366f1"
          description={models?.zero_dce?.device ? `Device: ${models.zero_dce.device}` : 'Zero-reference image enhancement model'}
        />
        <MetricCard
          title="YOLOv8 Status"
          value={<StatusBadge status={yoloStatus} label={yoloStatus === 'loaded' ? 'Loaded' : yoloStatus === 'error' ? 'Not Loaded' : 'Loading…'} />}
          icon={<Eye size={18} />}
          color="#3b82f6"
          description={models?.yolov8?.model_name ? `Model: ${models.yolov8.model_name}` : 'Object detection model'}
        />
        <MetricCard
          title="Compute Device"
          value={loading ? <LoadingSpinner size={20} /> : device}
          icon={<Cpu size={18} />}
          color="#10b981"
          description="Inference hardware"
        />
        <MetricCard
          title="Images Processed"
          value={loading ? '—' : processed}
          icon={<Image size={18} />}
          color="#f59e0b"
          description="Session history entries"
        />
      </div>

      {/* Recent History */}
      <div className="card" style={{ marginBottom:'1.5rem' }}>
        <div className="section-title">
          <Clock size={18} /> Recent History
        </div>
        {loading ? (
          <LoadingSpinner text="Loading history…" />
        ) : history.length === 0 ? (
          <div style={{ textAlign:'center', padding:'2rem', color:'var(--text-muted)' }}>
            <Image size={32} opacity={0.4} style={{ marginBottom:'.5rem' }} />
            <p style={{ fontSize:'.875rem' }}>No processing history yet. Enhance an image to get started.</p>
          </div>
        ) : (
          <div style={{ overflowX:'auto' }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Filename</th>
                  <th>Degradation</th>
                  <th>Severity</th>
                  <th>Strategy</th>
                  <th>Time (ms)</th>
                  <th>Timestamp</th>
                </tr>
              </thead>
              <tbody>
                {history.slice(0, 20).map((row, i) => (
                  <tr key={row.id ?? i}>
                    <td style={{ color:'var(--text-muted)' }}>{i + 1}</td>
                    <td style={{ fontFamily:'monospace', fontSize:'.8rem' }}>{row.filename ?? '—'}</td>
                    <td><StatusBadge status="blue" label={row.degradation_type ?? '—'} /></td>
                    <td><StatusBadge status={(row.severity_level ?? '').toLowerCase()} label={row.severity_level ?? '—'} /></td>
                    <td style={{ fontSize:'.8rem', color:'var(--text-secondary)' }}>{row.strategy ?? '—'}</td>
                    <td>{row.processing_time_ms != null ? `${row.processing_time_ms}` : '—'}</td>
                    <td style={{ fontSize:'.78rem', color:'var(--text-muted)' }}>{row.timestamp ? new Date(row.timestamp).toLocaleString() : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Capabilities */}
      <div>
        <div className="section-title">Key Capabilities</div>
        <div className="grid-4">
          {capabilities.map(({ icon: Icon, color, title, desc }) => (
            <div key={title} className="card" style={{ borderTop:`2px solid ${color}` }}>
              <div style={{ width:40, height:40, borderRadius:10, background:`${color}20`, display:'flex', alignItems:'center', justifyContent:'center', marginBottom:'.85rem' }}>
                <Icon size={20} color={color} />
              </div>
              <h3 style={{ fontWeight:700, fontSize:'.9rem', marginBottom:'.4rem' }}>{title}</h3>
              <p style={{ color:'var(--text-secondary)', fontSize:'.82rem', lineHeight:1.6 }}>{desc}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
