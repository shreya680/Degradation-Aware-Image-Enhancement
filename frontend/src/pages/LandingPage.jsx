import React from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Sparkles, Sun, Zap, Layers, Archive,
  ArrowRight, Database, Cpu, Eye, Shield,
  ChevronRight, Star, GitBranch, Activity,
} from 'lucide-react';

const degradationTypes = [
  { icon: Sun,     title: 'Low-Light',    color: '#f59e0b', desc: 'Images captured in poor lighting conditions with insufficient exposure.' },
  { icon: Zap,     title: 'Noise',        color: '#3b82f6', desc: 'Sensor noise, random variation in pixel values disrupting clarity.' },
  { icon: Layers,  title: 'Blur',         color: '#8b5cf6', desc: 'Motion blur or defocus reducing sharpness and edge definition.' },
  { icon: Archive, title: 'Compression',  color: '#10b981', desc: 'Lossy compression artifacts such as JPEG blocking and ringing.' },
];

const steps = [
  { n:'01', title:'Upload Image',      desc:'Drag & drop or browse to upload a degraded image (JPEG/PNG).' },
  { n:'02', title:'Auto-Analyze',      desc:'Signal processing analysis detects degradation type and severity automatically.' },
  { n:'03', title:'AI Enhancement',    desc:'Zero-DCE model and adaptive controller restore image quality perceptually.' },
  { n:'04', title:'Evaluate & Detect', desc:'Run YOLOv8 object detection on both images, compare PSNR/SSIM/LPIPS metrics.' },
];

const datasets = [
  { name: 'LOL Dataset',   desc: 'Low-light object localization — 500 paired images (485 train / 15 eval)', color: '#6366f1' },
  { name: 'ExDark Dataset',desc: '12-category dark image dataset for extreme low-light object detection',    color: '#3b82f6' },
];

const applications = [
  { icon: Eye,    title: 'Surveillance',     desc: 'Enhanced night-vision footage for security cameras.' },
  { icon: Cpu,    title: 'Autonomous Vehicles', desc: 'Better scene understanding in adverse conditions.' },
  { icon: Shield, title: 'Medical Imaging',  desc: 'Clearer diagnostic images from low-quality scans.' },
  { icon: Star,   title: 'Photography',      desc: 'Consumer-grade photo enhancement with AI quality.' },
];

export default function LandingPage() {
  const navigate = useNavigate();

  return (
    <div style={{ minHeight: '100vh', background: 'var(--bg-primary)', overflowX: 'hidden' }}>

      {/* ── Hero ── */}
      <section style={{ position: 'relative', minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden' }}>
        {/* Animated gradient bg */}
        <div style={{
          position: 'absolute', inset: 0,
          background: 'radial-gradient(ellipse 80% 60% at 50% 0%, rgba(99,102,241,0.18) 0%, transparent 70%), radial-gradient(ellipse 60% 50% at 80% 80%, rgba(59,130,246,0.12) 0%, transparent 60%), var(--bg-primary)',
        }} />
        {/* Decorative blobs */}
        <div style={{ position:'absolute', top:'10%', left:'5%',  width:300, height:300, borderRadius:'50%', background:'rgba(99,102,241,0.08)', filter:'blur(60px)', pointerEvents:'none' }} />
        <div style={{ position:'absolute', bottom:'10%', right:'5%', width:400, height:400, borderRadius:'50%', background:'rgba(59,130,246,0.07)', filter:'blur(80px)', pointerEvents:'none' }} />

        <div style={{ position:'relative', textAlign:'center', maxWidth:820, margin:'0 auto', padding:'2rem 1.5rem' }}>
          <div className="badge badge-violet" style={{ display:'inline-flex', marginBottom:'1.5rem', fontSize:'0.8rem' }}>
            <Sparkles size={13} /> AI/ML Research Platform
          </div>

          <h1 style={{
            fontSize: 'clamp(2rem, 5vw, 3.75rem)', fontWeight: 900, lineHeight: 1.1, marginBottom: '1.25rem',
            background: 'linear-gradient(135deg, #f1f5f9 30%, #6366f1 70%, #3b82f6)',
            WebkitBackgroundClip:'text', WebkitTextFillColor:'transparent', backgroundClip:'text',
          }}>
            Degradation-Aware<br />Image Enhancement
          </h1>

          <p style={{ fontSize: 'clamp(1rem, 2vw, 1.2rem)', color: 'var(--text-secondary)', maxWidth: 600, margin: '0 auto 2.5rem', lineHeight: 1.7 }}>
            Perceptual Quality &amp; Downstream Task Preservation using Zero-DCE, YOLOv8 and adaptive signal-aware controllers.
          </p>

          <div style={{ display:'flex', gap:'1rem', justifyContent:'center', flexWrap:'wrap' }}>
            <button className="btn btn-primary" onClick={() => navigate('/enhance')} style={{ padding:'0.85rem 2rem', fontSize:'1rem' }}>
              <Sparkles size={18} /> Get Started <ArrowRight size={16} />
            </button>
            <button className="btn btn-secondary" onClick={() => navigate('/')} style={{ padding:'0.85rem 2rem', fontSize:'1rem' }}>
              View Dashboard <ChevronRight size={16} />
            </button>
          </div>
        </div>
      </section>

      <div style={{ maxWidth: 1100, margin: '0 auto', padding: '0 1.5rem 4rem' }}>

        {/* ── Problem Statement ── */}
        <section style={{ marginBottom: '5rem' }}>
          <div style={{ textAlign:'center', marginBottom:'2.5rem' }}>
            <h2 style={{ fontSize:'2rem', fontWeight:800, color:'var(--text-primary)', marginBottom:'.5rem' }}>The Problem</h2>
            <p style={{ color:'var(--text-secondary)', maxWidth:600, margin:'0 auto' }}>
              Real-world images suffer from multiple degradation types that reduce both perceptual quality and downstream AI task performance.
            </p>
          </div>
          <div className="grid-2" style={{ gap:'1.5rem' }}>
            <div className="card glass-panel">
              <div style={{ display:'flex', alignItems:'center', gap:'.75rem', marginBottom:'.85rem' }}>
                <div style={{ width:36, height:36, borderRadius:8, background:'rgba(239,68,68,0.15)', display:'flex', alignItems:'center', justifyContent:'center' }}>
                  <Activity size={18} color="#ef4444" />
                </div>
                <h3 style={{ fontWeight:700, fontSize:'1rem' }}>Quality Degradation</h3>
              </div>
              <p style={{ color:'var(--text-secondary)', fontSize:'.9rem', lineHeight:1.7 }}>
                Low-light, noise, blur and compression artefacts severely degrade image quality, making automated analysis unreliable. Standard metrics like PSNR and SSIM fail to capture perceptual fidelity.
              </p>
            </div>
            <div className="card glass-panel">
              <div style={{ display:'flex', alignItems:'center', gap:'.75rem', marginBottom:'.85rem' }}>
                <div style={{ width:36, height:36, borderRadius:8, background:'rgba(245,158,11,0.15)', display:'flex', alignItems:'center', justifyContent:'center' }}>
                  <Eye size={18} color="#f59e0b" />
                </div>
                <h3 style={{ fontWeight:700, fontSize:'1rem' }}>Detection Failure</h3>
              </div>
              <p style={{ color:'var(--text-secondary)', fontSize:'.9rem', lineHeight:1.7 }}>
                Object detectors such as YOLOv8 suffer significant mAP drops on degraded images. Enhancement methods that maximise PSNR often <em>worsen</em> detection accuracy — a critical trade-off.
              </p>
            </div>
          </div>
        </section>

        {/* ── Proposed Solution ── */}
        <section style={{ marginBottom: '5rem' }}>
          <div style={{ textAlign:'center', marginBottom:'2.5rem' }}>
            <h2 style={{ fontSize:'2rem', fontWeight:800, color:'var(--text-primary)', marginBottom:'.5rem' }}>Proposed Solution</h2>
          </div>
          <div className="card" style={{ borderColor:'rgba(99,102,241,0.3)', borderLeft:'4px solid #6366f1' }}>
            <div className="grid-3" style={{ gap:'1.5rem' }}>
              {[
                { icon: GitBranch, color:'#6366f1', title:'Degradation-Aware Routing', desc:'Automatic detection of degradation type and severity guides which enhancement strategy is applied — preventing over-processing.' },
                { icon: Sparkles,  color:'#3b82f6', title:'Zero-DCE Enhancement',      desc:'Zero-Reference Deep Curve Estimation enhances low-light images without paired training data, using light-enhancement curves.' },
                { icon: Shield,    color:'#10b981', title:'Task-Preservation Focus',    desc:'Enhancement is evaluated not just on PSNR/SSIM but also on downstream YOLOv8 object detection performance.' },
              ].map(({ icon: Icon, color, title, desc }) => (
                <div key={title} style={{ textAlign:'center' }}>
                  <div style={{ width:52, height:52, borderRadius:12, background:`rgba(${color === '#6366f1' ? '99,102,241' : color === '#3b82f6' ? '59,130,246' : '16,185,129'},0.15)`, display:'flex', alignItems:'center', justifyContent:'center', margin:'0 auto 1rem' }}>
                    <Icon size={22} color={color} />
                  </div>
                  <h3 style={{ fontWeight:700, marginBottom:'.5rem' }}>{title}</h3>
                  <p style={{ color:'var(--text-secondary)', fontSize:'.875rem', lineHeight:1.6 }}>{desc}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* ── How It Works ── */}
        <section style={{ marginBottom: '5rem' }}>
          <div style={{ textAlign:'center', marginBottom:'2.5rem' }}>
            <h2 style={{ fontSize:'2rem', fontWeight:800, color:'var(--text-primary)', marginBottom:'.5rem' }}>How It Works</h2>
          </div>
          <div className="grid-4" style={{ gap:'1rem' }}>
            {steps.map((s, i) => (
              <div key={s.n} className="card" style={{ position:'relative', paddingTop:'1.75rem' }}>
                <div style={{ position:'absolute', top:'-14px', left:'1rem', width:28, height:28, borderRadius:7, background:'linear-gradient(135deg,#3b82f6,#6366f1)', display:'flex', alignItems:'center', justifyContent:'center', fontSize:'.7rem', fontWeight:800, color:'#fff' }}>
                  {s.n}
                </div>
                {i < steps.length - 1 && (
                  <ChevronRight size={16} style={{ position:'absolute', top:'50%', right:-12, transform:'translateY(-50%)', color:'var(--text-muted)', zIndex:1 }} />
                )}
                <h3 style={{ fontWeight:700, fontSize:'.95rem', marginBottom:'.5rem' }}>{s.title}</h3>
                <p style={{ color:'var(--text-secondary)', fontSize:'.82rem', lineHeight:1.6 }}>{s.desc}</p>
              </div>
            ))}
          </div>
        </section>

        {/* ── Degradation Types ── */}
        <section style={{ marginBottom: '5rem' }}>
          <div style={{ textAlign:'center', marginBottom:'2.5rem' }}>
            <h2 style={{ fontSize:'2rem', fontWeight:800, color:'var(--text-primary)', marginBottom:'.5rem' }}>Supported Degradation Types</h2>
          </div>
          <div className="grid-4" style={{ gap:'1rem' }}>
            {degradationTypes.map(({ icon: Icon, title, color, desc }) => (
              <div key={title} className="card" style={{ textAlign:'center', borderTop:`2px solid ${color}` }}>
                <div style={{ width:48, height:48, borderRadius:12, background:`${color}20`, display:'flex', alignItems:'center', justifyContent:'center', margin:'0 auto 1rem' }}>
                  <Icon size={24} color={color} />
                </div>
                <h3 style={{ fontWeight:700, marginBottom:'.5rem', fontSize:'.95rem' }}>{title}</h3>
                <p style={{ color:'var(--text-secondary)', fontSize:'.82rem', lineHeight:1.6 }}>{desc}</p>
              </div>
            ))}
          </div>
        </section>

        {/* ── Datasets ── */}
        <section style={{ marginBottom: '5rem' }}>
          <div style={{ textAlign:'center', marginBottom:'2.5rem' }}>
            <h2 style={{ fontSize:'2rem', fontWeight:800, color:'var(--text-primary)', marginBottom:'.5rem' }}>Datasets</h2>
          </div>
          <div className="grid-2" style={{ gap:'1.5rem' }}>
            {datasets.map(d => (
              <div key={d.name} className="card" style={{ borderLeft:`3px solid ${d.color}` }}>
                <div style={{ display:'flex', alignItems:'center', gap:'.75rem', marginBottom:'.75rem' }}>
                  <Database size={20} color={d.color} />
                  <h3 style={{ fontWeight:700 }}>{d.name}</h3>
                </div>
                <p style={{ color:'var(--text-secondary)', fontSize:'.875rem' }}>{d.desc}</p>
              </div>
            ))}
          </div>
        </section>

        {/* ── Applications ── */}
        <section style={{ marginBottom: '4rem' }}>
          <div style={{ textAlign:'center', marginBottom:'2.5rem' }}>
            <h2 style={{ fontSize:'2rem', fontWeight:800, color:'var(--text-primary)', marginBottom:'.5rem' }}>Applications</h2>
          </div>
          <div className="grid-4" style={{ gap:'1rem' }}>
            {applications.map(({ icon: Icon, title, desc }) => (
              <div key={title} className="card" style={{ textAlign:'center' }}>
                <Icon size={28} style={{ color:'var(--accent-blue)', marginBottom:'.75rem' }} />
                <h3 style={{ fontWeight:700, fontSize:'.95rem', marginBottom:'.4rem' }}>{title}</h3>
                <p style={{ color:'var(--text-secondary)', fontSize:'.82rem', lineHeight:1.6 }}>{desc}</p>
              </div>
            ))}
          </div>
        </section>

        {/* ── CTA ── */}
        <section style={{ textAlign:'center' }}>
          <div className="card glass-panel" style={{ padding:'3rem 2rem', borderColor:'rgba(99,102,241,0.2)' }}>
            <h2 style={{ fontSize:'1.75rem', fontWeight:800, marginBottom:'.75rem' }}>Ready to Enhance?</h2>
            <p style={{ color:'var(--text-secondary)', marginBottom:'1.75rem' }}>Upload a degraded image and see the AI enhancement in action.</p>
            <button className="btn btn-primary" onClick={() => navigate('/enhance')} style={{ padding:'.85rem 2.5rem', fontSize:'1rem' }}>
              <Sparkles size={18} /> Start Enhancing <ArrowRight size={16} />
            </button>
          </div>
        </section>
      </div>
    </div>
  );
}
