import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Database,
  ExternalLink,
  FolderOpen,
  CheckCircle,
  AlertCircle,
  Layers,
  FileImage,
  RefreshCw,
  Info,
  ArrowRight,
  BookOpen,
} from 'lucide-react';
import { getDatasetsStatus } from '../services/api.js';

export default function DatasetPage() {
  const navigate = useNavigate();
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [lolPath, setLolPath] = useState('data/LOL');
  const [exdarkPath, setExdarkPath] = useState('data/ExDark');

  const fetchStats = async () => {
    setLoading(true);
    try {
      const res = await getDatasetsStatus();
      setStats(res.data);
    } catch (err) {
      console.warn('Dataset status fetch error:', err);
      // Fallback local status
      setStats({
        lol: { available: false, path: 'data/LOL', total_pairs: 0, train_count: 0, eval_count: 0 },
        exdark: { available: false, path: 'data/ExDark', total_images: 0, categories_found: 0 },
      });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStats();
  }, []);

  return (
    <div className="page-wrapper" style={{ padding: '2rem', maxWidth: 1400, margin: '0 auto' }}>
      {/* Header */}
      <div className="page-header" style={{ marginBottom: '2rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
          <div style={{
            background: 'linear-gradient(135deg, #8b5cf6, #3b82f6)',
            padding: '0.6rem',
            borderRadius: '12px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}>
            <Database size={24} color="#fff" />
          </div>
          <div>
            <h1 style={{ fontSize: '1.75rem', fontWeight: 700, margin: 0 }}>Benchmark Datasets & Academic Integration</h1>
            <p style={{ color: 'var(--text-secondary)', margin: 0, fontSize: '0.9rem' }}>
              Rigorous paired reference assessment (LOL) and real-world downstream object detection (ExDark).
            </p>
          </div>
        </div>
      </div>

      {/* Dataset Cards Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(500px, 1fr))', gap: '2rem', marginBottom: '2.5rem' }}>
        
        {/* Dataset 1: LOL */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1rem' }}>
              <div>
                <span style={{
                  background: 'rgba(59, 130, 246, 0.15)',
                  color: 'var(--accent-blue)',
                  padding: '0.25rem 0.6rem',
                  borderRadius: '6px',
                  fontSize: '0.75rem',
                  fontWeight: 600,
                  textTransform: 'uppercase',
                }}>
                  Paired Image Enhancement
                </span>
                <h2 style={{ fontSize: '1.35rem', fontWeight: 700, margin: '0.5rem 0 0.2rem' }}>LOL Dataset (v1)</h2>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>BMVC 2018 (Wei et al.)</div>
              </div>

              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.4rem',
                fontSize: '0.8rem',
                background: stats?.lol?.available ? 'rgba(16, 185, 129, 0.1)' : 'rgba(245, 158, 11, 0.1)',
                color: stats?.lol?.available ? 'var(--accent-green)' : 'var(--accent-yellow)',
                padding: '0.35rem 0.75rem',
                borderRadius: '8px',
                fontWeight: 600,
              }}>
                {stats?.lol?.available ? <CheckCircle size={14} /> : <AlertCircle size={14} />}
                {stats?.lol?.available ? 'Configured & Active' : 'Directory Ready (Optional Download)'}
              </div>
            </div>

            <p style={{ color: 'var(--text-secondary)', fontSize: '0.88rem', lineHeight: 1.6, marginBottom: '1.25rem' }}>
              The Low-Light (LOL) dataset contains 500 low/normal-light image pairs captured in real-world scenes by changing exposure time and ISO. It is specifically designed for supervised training, validation, and reference-based perceptual image quality assessment (PSNR, SSIM, LPIPS).
            </p>

            <div style={{ background: 'rgba(0,0,0,0.2)', padding: '1rem', borderRadius: '8px', marginBottom: '1.25rem' }}>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem', textAlign: 'center' }}>
                <div>
                  <div style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--text-primary)' }}>485</div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Training Pairs (our485)</div>
                </div>
                <div>
                  <div style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--text-primary)' }}>15</div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Evaluation Pairs (eval15)</div>
                </div>
                <div>
                  <div style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--accent-blue)' }}>500</div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Total Paired Images</div>
                </div>
              </div>
            </div>

            <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginBottom: '1rem' }}>
              <strong>Expected Folder Structure:</strong>
              <pre style={{
                background: '#070a12',
                padding: '0.6rem 0.8rem',
                borderRadius: '6px',
                marginTop: '0.3rem',
                fontSize: '0.75rem',
                color: '#94a3b8',
                overflowX: 'auto',
              }}>
                {`data/LOL/
  our485/
    low/     (485 images)
    high/    (485 ground truth)
  eval15/
    low/     (15 test images)
    high/    (15 ground truth)`}
              </pre>
            </div>
          </div>

          <div style={{ display: 'flex', gap: '0.75rem', marginTop: '1rem' }}>
            <a
              href="https://daooshee.github.io/BMVC2018website/"
              target="_blank"
              rel="noreferrer"
              className="btn btn-secondary"
              style={{ fontSize: '0.8rem', flex: 1, justifyContent: 'center' }}
            >
              <ExternalLink size={14} /> Official LOL Webpage
            </a>
          </div>
        </div>

        {/* Dataset 2: ExDark */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1rem' }}>
              <div>
                <span style={{
                  background: 'rgba(139, 92, 246, 0.15)',
                  color: 'var(--accent-purple)',
                  padding: '0.25rem 0.6rem',
                  borderRadius: '6px',
                  fontSize: '0.75rem',
                  fontWeight: 600,
                  textTransform: 'uppercase',
                }}>
                  Downstream Object Detection
                </span>
                <h2 style={{ fontSize: '1.35rem', fontWeight: 700, margin: '0.5rem 0 0.2rem' }}>ExDark Dataset</h2>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>CVIU 2020 (Loh & Chan)</div>
              </div>

              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.4rem',
                fontSize: '0.8rem',
                background: stats?.exdark?.available ? 'rgba(16, 185, 129, 0.1)' : 'rgba(245, 158, 11, 0.1)',
                color: stats?.exdark?.available ? 'var(--accent-green)' : 'var(--accent-yellow)',
                padding: '0.35rem 0.75rem',
                borderRadius: '8px',
                fontWeight: 600,
              }}>
                {stats?.exdark?.available ? <CheckCircle size={14} /> : <AlertCircle size={14} />}
                {stats?.exdark?.available ? 'Configured & Active' : 'Directory Ready (Optional Download)'}
              </div>
            </div>

            <p style={{ color: 'var(--text-secondary)', fontSize: '0.88rem', lineHeight: 1.6, marginBottom: '1.25rem' }}>
              The Exclusively Dark (ExDark) dataset is specifically dedicated to low-light object detection. It spans very low-light outdoor and indoor environments across 12 distinct categories. <strong>Crucial academic distinction:</strong> ExDark does NOT have paired normal-light references; it evaluates downstream task preservation.
            </p>

            <div style={{ background: 'rgba(0,0,0,0.2)', padding: '1rem', borderRadius: '8px', marginBottom: '1.25rem' }}>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem', textAlign: 'center' }}>
                <div>
                  <div style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--text-primary)' }}>7,363</div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Low-Light Images</div>
                </div>
                <div>
                  <div style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--text-primary)' }}>12</div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Object Classes</div>
                </div>
                <div>
                  <div style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--accent-purple)' }}>COCO/VOC</div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Bounding Box Format</div>
                </div>
              </div>
            </div>

            <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginBottom: '1rem' }}>
              <strong>12 Supported Classes:</strong>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem', marginTop: '0.4rem' }}>
                {['Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat', 'Chair', 'Cup', 'Dog', 'Motorbike', 'People', 'Table'].map((cls) => (
                  <span key={cls} style={{
                    background: 'rgba(255,255,255,0.06)',
                    padding: '0.2rem 0.5rem',
                    borderRadius: '4px',
                    fontSize: '0.75rem',
                  }}>
                    {cls}
                  </span>
                ))}
              </div>
            </div>
          </div>

          <div style={{ display: 'flex', gap: '0.75rem', marginTop: '1rem' }}>
            <a
              href="https://github.com/cs-chan/Exclusively-Dark-Image-Dataset"
              target="_blank"
              rel="noreferrer"
              className="btn btn-secondary"
              style={{ fontSize: '0.8rem', flex: 1, justifyContent: 'center' }}
            >
              <ExternalLink size={14} /> Official ExDark GitHub
            </a>
          </div>
        </div>
      </div>

      {/* Dataset Setup Instructions Card */}
      <div className="card" style={{ marginBottom: '2rem' }}>
        <h3 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <FolderOpen size={18} color="var(--accent-blue)" /> Windows Local Setup Instructions
        </h3>
        <p style={{ fontSize: '0.88rem', color: 'var(--text-secondary)', lineHeight: 1.6, marginBottom: '1rem' }}>
          Due to dataset file sizes (hundreds of megabytes), raw datasets are never bundled directly in Git repositories. The application automatically detects dataset images placed into the designated project folders:
        </p>

        <ol style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: 1.8, paddingLeft: '1.25rem' }}>
          <li>Download LOL-v1 from the official link above and unzip the folders into <code style={{ color: 'var(--text-primary)' }}>data\LOL\our485\</code> and <code style={{ color: 'var(--text-primary)' }}>data\LOL\eval15\</code>.</li>
          <li>Download the ExDark dataset and place the 12 class subdirectories into <code style={{ color: 'var(--text-primary)' }}>data\ExDark\</code>.</li>
          <li>Click the button below to re-scan directories and refresh detection status.</li>
        </ol>

        <div style={{ marginTop: '1.25rem', display: 'flex', gap: '0.75rem' }}>
          <button className="btn btn-secondary" onClick={fetchStats} disabled={loading}>
            <RefreshCw size={14} className={loading ? 'spin' : ''} /> Rescan Datasets
          </button>
          <button className="btn btn-primary" onClick={() => navigate('/enhance')}>
            Test Adaptive Enhancement <ArrowRight size={14} />
          </button>
        </div>
      </div>
    </div>
  );
}
