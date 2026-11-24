import React from 'react';
import { CheckCircle, XCircle, Loader } from 'lucide-react';

const configs = {
  loaded:  { cls: 'badge-green',  icon: <CheckCircle size={12} />, defaultLabel: 'Loaded'  },
  ready:   { cls: 'badge-green',  icon: <CheckCircle size={12} />, defaultLabel: 'Ready'   },
  error:   { cls: 'badge-red',    icon: <XCircle size={12} />,     defaultLabel: 'Error'   },
  offline: { cls: 'badge-red',    icon: <XCircle size={12} />,     defaultLabel: 'Offline' },
  loading: { cls: 'badge-yellow', icon: <Loader size={12} style={{animation:'spin 1s linear infinite'}} />, defaultLabel: 'Loading' },
  pending: { cls: 'badge-yellow', icon: <Loader size={12} />,      defaultLabel: 'Pending' },
  low:     { cls: 'badge-green',  icon: null, defaultLabel: 'Low'    },
  medium:  { cls: 'badge-yellow', icon: null, defaultLabel: 'Medium' },
  high:    { cls: 'badge-red',    icon: null, defaultLabel: 'High'   },
};

export default function StatusBadge({ status = 'loading', label }) {
  const key = (status || '').toLowerCase();
  const cfg = configs[key] || { cls: 'badge-blue', icon: null, defaultLabel: status };
  return (
    <span className={`badge ${cfg.cls}`}>
      {cfg.icon}
      {label ?? cfg.defaultLabel}
    </span>
  );
}
