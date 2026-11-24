import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  Sparkles,
  ScanSearch,
  BarChart3,
  Database,
  Info,
  Cpu,
  Wifi,
  WifiOff,
} from 'lucide-react';
import { getHealth } from '../services/api.js';
import { useEffect, useState } from 'react';

const navItems = [
  { label: 'Dashboard',            icon: LayoutDashboard, to: '/'          },
  { label: 'Image Enhancement',    icon: Sparkles,         to: '/enhance'   },
  { label: 'Object Detection',     icon: ScanSearch,       to: '/detect'    },
  { label: 'Performance Analytics',icon: BarChart3,        to: '/analytics' },
  { label: 'Dataset Information',  icon: Database,         to: '/datasets'  },
  { label: 'About Project',        icon: Info,             to: '/about'     },
];

export default function Sidebar() {
  const [health, setHealth]   = useState(null);
  const [online, setOnline]   = useState(true);

  useEffect(() => {
    let alive = true;
    const fetchHealth = async () => {
      try {
        const res = await getHealth();
        if (alive) { setHealth(res.data); setOnline(true); }
      } catch {
        if (alive) setOnline(false);
      }
    };
    fetchHealth();
    const id = setInterval(fetchHealth, 30000);
    return () => { alive = false; clearInterval(id); };
  }, []);

  return (
    <aside className="sidebar">
      {/* Logo */}
      <div className="sidebar-logo">
        <div className="logo-mark">
          <div className="logo-icon">D</div>
          <div className="logo-text">
            <h2>DAIE System</h2>
            <span>AI · Enhancement</span>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="sidebar-nav">
        <div className="nav-label">Navigation</div>
        {navItems.map(({ label, icon: Icon, to }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) => `nav-item${isActive ? ' active' : ''}`}
          >
            <Icon className="nav-icon" size={18} />
            <span>{label}</span>
          </NavLink>
        ))}
      </nav>

      {/* Footer / status */}
      <div className="sidebar-footer">
        <div className="status-indicator">
          <div className="status-row">
            <span className="status-label">API</span>
            <span className="status-value" style={{ display:'flex', alignItems:'center', gap:4 }}>
              {online ? (
                <><span className="status-dot green" /><Wifi size={12} style={{color:'#10b981'}} /></>
              ) : (
                <><span className="status-dot red" /><WifiOff size={12} style={{color:'#ef4444'}} /></>
              )}
            </span>
          </div>
          <div className="status-row">
            <span className="status-label">Device</span>
            <span className="status-value" style={{ display:'flex', alignItems:'center', gap:4 }}>
              <Cpu size={11} />
              {health?.device ?? '—'}
            </span>
          </div>
          <div className="status-row">
            <span className="status-label">ZeroDCE</span>
            <span className="status-value">
              {health == null ? '—' : (health.models_loaded?.zero_dce ? '✓ loaded' : '✗ off')}
            </span>
          </div>
          <div className="status-row">
            <span className="status-label">YOLOv8</span>
            <span className="status-value">
              {health == null ? '—' : (health.models_loaded?.yolov8 ? '✓ loaded' : '✗ off')}
            </span>
          </div>
        </div>
      </div>
    </aside>
  );
}
