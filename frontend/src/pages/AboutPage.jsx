import React from 'react';
import { useNavigate } from 'react-router-dom';
import {
  GraduationCap,
  Layers,
  Cpu,
  Eye,
  CheckCircle2,
  Sparkles,
  Shield,
  Activity,
  ArrowRight,
  Database,
  Code2,
  FileCheck,
  Zap,
} from 'lucide-react';

export default function AboutPage() {
  const navigate = useNavigate();

  return (
    <div className="page-wrapper" style={{ padding: '2rem', maxWidth: 1400, margin: '0 auto' }}>
      {/* Title */}
      <div className="page-header" style={{ marginBottom: '2.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
          <div style={{
            background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
            padding: '0.6rem',
            borderRadius: '12px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}>
            <GraduationCap size={26} color="#fff" />
          </div>
          <div>
            <h1 style={{ fontSize: '1.85rem', fontWeight: 800, margin: 0 }}>
              Degradation-Aware Image Enhancement with Perceptual Quality and Downstream Task Preservation
            </h1>
            <p style={{ color: 'var(--text-secondary)', margin: 0, fontSize: '0.95rem' }}>
              Academic Research Project – SIC AIML Evaluation &amp; Computer Vision Architecture
            </p>
          </div>
        </div>
      </div>

      {/* Research Abstract */}
      <div className="card" style={{ marginBottom: '2rem', borderLeft: '4px solid var(--accent-blue)' }}>
        <h2 style={{ fontSize: '1.2rem', fontWeight: 700, marginBottom: '0.75rem', color: 'var(--accent-blue)' }}>
          Research Abstract &amp; Problem Statement
        </h2>
        <p style={{ color: 'var(--text-secondary)', lineHeight: 1.7, fontSize: '0.92rem', marginBottom: '1rem' }}>
          Images captured in wild environments frequently suffer from multifaceted degradations including severe low-light, Gaussian/sensor noise, motion blur, and JPEG compression artifacts. Conventional image enhancement frameworks apply static global filters or unguided brightness stretching, which invariably induce color distortions, amplify sensor noise, or blow out highlights. Crucially, conventional metrics exclusively focus on human visual aesthetics while neglecting downstream computer vision performance.
        </p>
        <p style={{ color: 'var(--text-secondary)', lineHeight: 1.7, fontSize: '0.92rem', margin: 0 }}>
          This project proposes a <strong>Degradation-Aware Adaptive Enhancement System</strong> that dynamically assesses degradation characteristics, formulates a tailored enhancement trajectory utilizing a zero-reference deep curve estimation network (<strong>Zero-DCE</strong>), and quantifies downstream task preservation through fine-grained object detection with <strong>YOLOv8</strong>.
        </p>
      </div>

      {/* 4 Core Pillars */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '1.25rem', marginBottom: '2.5rem' }}>
        <div className="card">
          <div style={{ color: 'var(--accent-yellow)', marginBottom: '0.5rem' }}><Zap size={22} /></div>
          <h3 style={{ fontSize: '1.05rem', fontWeight: 700, marginBottom: '0.4rem' }}>1. Degradation Detection</h3>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
            Automated signal-processing analysis evaluating mean luminescence, Laplacian high-frequency variance, and block DCT discontinuities to classify severity into Low, Medium, and High.
          </p>
        </div>

        <div className="card">
          <div style={{ color: 'var(--accent-green)', marginBottom: '0.5rem' }}><Sparkles size={22} /></div>
          <h3 style={{ fontSize: '1.05rem', fontWeight: 700, marginBottom: '0.4rem' }}>2. Adaptive Enhancement</h3>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
            Dynamic controller that dynamically parameterizes Zero-DCE curve iterations or triggers complementary filtering, avoiding over-enhancement and preserving natural tonal dynamic range.
          </p>
        </div>

        <div className="card">
          <div style={{ color: 'var(--accent-blue)', marginBottom: '0.5rem' }}><Activity size={22} /></div>
          <h3 style={{ fontSize: '1.05rem', fontWeight: 700, marginBottom: '0.4rem' }}>3. Perceptual Quality</h3>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
            Multi-metric evaluation integrating Peak Signal-to-Noise Ratio (PSNR), Structural Similarity (SSIM), and Deep Feature Distance (LPIPS) validated against paired LOL ground truth.
          </p>
        </div>

        <div className="card">
          <div style={{ color: 'var(--accent-purple)', marginBottom: '0.5rem' }}><Eye size={22} /></div>
          <h3 style={{ fontSize: '1.05rem', fontWeight: 700, marginBottom: '0.4rem' }}>4. Downstream Preservation</h3>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
            Validating that enhancement actively aids downstream vision models (YOLOv8) by improving object discoverability and confidence across 80 COCO categories and ExDark scenes.
          </p>
        </div>
      </div>

      {/* System Pipeline Architecture */}
      <div className="card" style={{ marginBottom: '2.5rem' }}>
        <h2 style={{ fontSize: '1.2rem', fontWeight: 700, marginBottom: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Layers size={20} color="var(--accent-violet)" /> End-to-End System Architecture
        </h2>

        <div style={{
          background: '#070a12',
          padding: '1.5rem',
          borderRadius: '12px',
          border: '1px solid rgba(255,255,255,0.06)',
          display: 'flex',
          flexDirection: 'column',
          gap: '1rem',
        }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '1rem', textAlign: 'center' }}>
            <div style={{ background: 'var(--bg-card)', padding: '1rem', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>STAGE 1</div>
              <strong style={{ fontSize: '0.9rem', color: 'var(--accent-blue)' }}>Input Image</strong>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>Degraded RGB (JPEG/PNG)</div>
            </div>

            <div style={{ background: 'var(--bg-card)', padding: '1rem', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>STAGE 2</div>
              <strong style={{ fontSize: '0.9rem', color: 'var(--accent-yellow)' }}>Degradation Analysis</strong>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>Luminance &amp; Laplacian</div>
            </div>

            <div style={{ background: 'var(--bg-card)', padding: '1rem', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>STAGE 3</div>
              <strong style={{ fontSize: '0.9rem', color: 'var(--accent-violet)' }}>Adaptive Selection</strong>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>Strategy &amp; Curves Config</div>
            </div>

            <div style={{ background: 'var(--bg-card)', padding: '1rem', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>STAGE 4</div>
              <strong style={{ fontSize: '0.9rem', color: 'var(--accent-green)' }}>Zero-DCE Inference</strong>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>7-Conv Deep Curve Est.</div>
            </div>

            <div style={{ background: 'var(--bg-card)', padding: '1rem', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>STAGE 5</div>
              <strong style={{ fontSize: '0.9rem', color: 'var(--accent-purple)' }}>YOLOv8 + Evaluation</strong>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>Preservation &amp; PSNR/SSIM</div>
            </div>
          </div>
        </div>
      </div>

      {/* Deep Learning Mathematical Formulation */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(450px, 1fr))', gap: '1.5rem', marginBottom: '2.5rem' }}>
        <div className="card">
          <h3 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Code2 size={18} color="var(--accent-green)" /> Zero-DCE Mathematical Formulation
          </h3>
          <p style={{ fontSize: '0.88rem', color: 'var(--text-secondary)', lineHeight: 1.6, marginBottom: '1rem' }}>
            Zero-DCE reformulates image enhancement as an estimation of higher-order Light-Enhancement (LE) curves. The transformation is recursively defined:
          </p>
          <div style={{
            background: '#070a12',
            padding: '1rem',
            borderRadius: '8px',
            fontFamily: 'monospace',
            color: 'var(--accent-green)',
            fontSize: '0.9rem',
            marginBottom: '1rem',
            textAlign: 'center',
          }}>
            LE_n(x) = LE_n-1(x) + A_n · LE_n-1(x)(1 - LE_n-1(x))
          </div>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: 1.6, margin: 0 }}>
            Where <code style={{ color: 'var(--text-primary)' }}>A_n</code> is a parameter map estimated by DCENet across 8 recursive iterations. The zero-reference loss formulation comprises spatial consistency, exposure control, color constancy, and illumination smoothness losses.
          </p>
        </div>

        <div className="card">
          <h3 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Shield size={18} color="var(--accent-purple)" /> Downstream Task Preservation (YOLOv8)
          </h3>
          <p style={{ fontSize: '0.88rem', color: 'var(--text-secondary)', lineHeight: 1.6, marginBottom: '1rem' }}>
            To rigorously confirm that visual enhancement does not hallucinate artifacts or obliterate structural cues, Ultralytics YOLOv8 runs identical inference on paired raw and enhanced images:
          </p>
          <ul style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: 1.7, paddingLeft: '1.25rem', margin: 0 }}>
            <li><strong>Confidence Parity:</strong> Identical confidence threshold (default 0.25) across both passes.</li>
            <li><strong>Class Distribution Delta:</strong> Tracks newly uncovered targets vs. preserved objects.</li>
            <li><strong>Zero Fabrication:</strong> Genuine tensor inference output without artificial score inflation.</li>
          </ul>
        </div>
      </div>

      {/* Technology Stack Grid */}
      <div className="card" style={{ marginBottom: '2.5rem' }}>
        <h2 style={{ fontSize: '1.2rem', fontWeight: 700, marginBottom: '1.25rem' }}>Complete Technology Stack</h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: '1rem' }}>
          {[
            { name: 'PyTorch', cat: 'Deep Learning', desc: 'DCENet & Tensor Ops' },
            { name: 'YOLOv8', cat: 'Object Detection', desc: 'Ultralytics Backbone' },
            { name: 'FastAPI', cat: 'Backend Framework', desc: 'Asynchronous REST APIs' },
            { name: 'OpenCV / PIL', cat: 'Computer Vision', desc: 'Image Pre/Post-processing' },
            { name: 'React 18 & Vite', cat: 'Frontend Architecture', desc: 'Modern Dark SaaS UI' },
            { name: 'Recharts', cat: 'Data Visualization', desc: 'Comparative Metrics Charts' },
            { name: 'Scikit-Image', cat: 'Quality Evaluation', desc: 'PSNR & SSIM Algorithms' },
            { name: 'SQLite3', cat: 'Metadata Persistence', desc: 'Processing History Log' },
          ].map((tech) => (
            <div key={tech.name} style={{ background: 'rgba(255,255,255,0.02)', padding: '0.85rem', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
              <div style={{ fontSize: '0.7rem', color: 'var(--accent-blue)', textTransform: 'uppercase', fontWeight: 600 }}>{tech.cat}</div>
              <strong style={{ fontSize: '0.95rem', display: 'block', margin: '0.2rem 0' }}>{tech.name}</strong>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{tech.desc}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Applications & Future Scope */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', marginBottom: '2.5rem' }}>
        <div className="card">
          <h3 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: '0.75rem' }}>Real-World Applications</h3>
          <ul style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: 1.8, paddingLeft: '1.25rem', margin: 0 }}>
            <li><strong>Autonomous Driving:</strong> Night-time object detection and pedestrian segmentation.</li>
            <li><strong>Surveillance &amp; Traffic:</strong> License plate and intruder tracking in low-illumination CCTV.</li>
            <li><strong>Medical Imaging:</strong> Low-dose X-ray and endoscopic luminescence normalization.</li>
            <li><strong>Remote Sensing:</strong> Satellite imagery contrast correction in shadowed terrain.</li>
          </ul>
        </div>

        <div className="card">
          <h3 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: '0.75rem' }}>Future Research Directions</h3>
          <ul style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: 1.8, paddingLeft: '1.25rem', margin: 0 }}>
            <li><strong>Unified Multi-Degradation Networks:</strong> Transformer-based restoration (Restormer/PromptIR).</li>
            <li><strong>Real-Time Video Enhancement:</strong> Temporal consistency constraints for 30+ FPS video.</li>
            <li><strong>Hardware Quantization:</strong> INT8 TensorRT / OpenVINO deployment for edge devices.</li>
          </ul>
        </div>
      </div>

      {/* Navigation Footer */}
      <div style={{ display: 'flex', justifyContent: 'center', marginTop: '2rem' }}>
        <button className="btn btn-primary" onClick={() => navigate('/enhance')} style={{ padding: '0.8rem 2rem', fontSize: '1rem' }}>
          Launch Image Enhancement Workspace <ArrowRight size={18} />
        </button>
      </div>
    </div>
  );
}
