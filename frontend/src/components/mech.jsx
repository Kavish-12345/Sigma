// Mechanical SVG pieces and animations: gears, gear train, car loader, piston, logo.
import { useEffect, useRef, useState } from "react";

// ---------- gear geometry ----------
export function gearPath(teeth, rOuter, rRoot) {
  const pts = [];
  const step = (Math.PI * 2) / teeth;
  for (let i = 0; i < teeth; i++) {
    const a = i * step;
    const t = [
      [rRoot, a],
      [rOuter, a + step * 0.18],
      [rOuter, a + step * 0.5],
      [rRoot, a + step * 0.68],
    ];
    t.forEach(([r, ang]) => pts.push(`${(r * Math.cos(ang)).toFixed(2)},${(r * Math.sin(ang)).toFixed(2)}`));
  }
  return `M${pts.join("L")}Z`;
}

export function Gear({ teeth = 12, r = 40, x = 0, y = 0, speed = 1, reverse = false, offset = 0, className = "", hub = true }) {
  // speed: seconds per turn scale; meshing gears turn at a rate inversely proportional to teeth
  const dur = (teeth / 12) * 6 * speed;
  const root = r - Math.max(4, r * 0.16);
  return (
    <g transform={`translate(${x} ${y})`} className={className}>
      <g
        className="gear-rot"
        style={{ animationDuration: `${dur}s`, animationDirection: reverse ? "reverse" : "normal" }}
      >
        <g transform={`rotate(${offset})`}>
          <path d={gearPath(teeth, r, root)} className="gear-body" />
          {hub && (
            <>
              <circle r={root * 0.62} className="gear-web" />
              <circle r={root * 0.22} className="gear-hub" />
              {[0, 60, 120, 180, 240, 300].map((d) => (
                <circle key={d} r={root * 0.09} cx={root * 0.42 * Math.cos((d * Math.PI) / 180)} cy={root * 0.42 * Math.sin((d * Math.PI) / 180)} className="gear-hole" />
              ))}
            </>
          )}
        </g>
      </g>
    </g>
  );
}

// ---------- hero gear train (three meshing gears) ----------
export function GearTrain({ className = "" }) {
  return (
    <svg viewBox="44 40 296 262" className={`geartrain ${className}`} aria-hidden="true">
      <Gear teeth={24} r={92} x={150} y={150} speed={1.2} />
      <Gear teeth={12} r={48} x={266.5} y={95.7} speed={1.2} reverse offset={8} className="accent" />
      <Gear teeth={16} r={62} x={266} y={231} speed={1.2} reverse offset={4} />
      <line x1="150" y1="150" x2="266.5" y2="95.7" className="center-line" />
      <line x1="150" y1="150" x2="266" y2="231" className="center-line" />
    </svg>
  );
}

// ---------- piston + crank (background) ----------
export function Piston({ className = "" }) {
  return (
    <svg viewBox="0 0 120 260" className={`piston ${className}`} aria-hidden="true">
      <rect x="22" y="10" width="76" height="150" className="cylinder" />
      <g className="piston-move">
        <rect x="28" y="40" width="64" height="44" rx="3" className="piston-head" />
        <line x1="28" y1="50" x2="92" y2="50" className="ring" />
        <line x1="28" y1="57" x2="92" y2="57" className="ring" />
        <line x1="60" y1="70" x2="60" y2="150" className="rod" />
      </g>
      <g transform="translate(60 205)">
        <g className="crank-rot">
          <circle r="38" className="crank-disc" />
          <circle r="5" cy="-22" className="crank-pin" />
          <circle r="7" className="gear-hub" />
        </g>
      </g>
    </svg>
  );
}

// ---------- ball bearing (balls orbit between the races) ----------
export function Bearing({ className = "" }) {
  const balls = Array.from({ length: 9 });
  return (
    <svg viewBox="-50 -50 100 100" className={`bearing ${className}`} aria-hidden="true">
      <circle r="46" className="race-outer" />
      <circle r="40" className="race-line" />
      <g>
        <animateTransform attributeName="transform" type="rotate" from="0 0 0" to="360 0 0" dur="6s" repeatCount="indefinite" />
        {balls.map((_, i) => {
          const a = (i / balls.length) * Math.PI * 2;
          return <circle key={i} r="5.2" cx={31 * Math.cos(a)} cy={31 * Math.sin(a)} className="ball" />;
        })}
      </g>
      <g>
        <animateTransform attributeName="transform" type="rotate" from="0 0 0" to="360 0 0" dur="3s" repeatCount="indefinite" />
        <circle r="22" className="race-inner" />
        <circle r="12" className="bore" />
        <rect x="-2" y="-16" width="4" height="5" className="key" />
      </g>
    </svg>
  );
}

// ---------- thinking loader: blueprint car driving ----------
export function CarLoader({ label = "Thinking" }) {
  return (
    <div className="car-loader" role="status" aria-label={label}>
      <svg viewBox="0 0 220 64" width="220" height="64" aria-hidden="true">
        <g className="road">
          {Array.from({ length: 12 }).map((_, i) => (
            <line key={i} x1={i * 24} y1="56" x2={i * 24 + 12} y2="56" />
          ))}
        </g>
        <g className="car">
          <path d="M18 40 L30 40 L42 24 L86 24 L102 38 L118 40 L120 48 L16 48 Z" className="car-body" />
          <path d="M46 27 L62 27 L62 38 L36 38 Z M66 27 L84 27 L96 38 L66 38 Z" className="car-window" />
          <g transform="translate(40 48)"><g className="wheel"><circle r="8" /><line x1="-8" y1="0" x2="8" y2="0" /><line x1="0" y1="-8" x2="0" y2="8" /></g></g>
          <g transform="translate(98 48)"><g className="wheel"><circle r="8" /><line x1="-8" y1="0" x2="8" y2="0" /><line x1="0" y1="-8" x2="0" y2="8" /></g></g>
          <circle cx="10" cy="44" r="3" className="puff p1" />
          <circle cx="4" cy="42" r="2" className="puff p2" />
        </g>
      </svg>
      <span className="loader-label">{label}</span>
    </div>
  );
}

// ---------- ranking loader: gear train with caption ----------
export function GearLoader({ label = "Ranking materials" }) {
  return (
    <div className="gear-loader" role="status" aria-label={label}>
      <svg viewBox="0 0 160 110" width="160" height="110" aria-hidden="true">
        <Gear teeth={16} r={34} x={52} y={58} speed={0.35} />
        <Gear teeth={10} r={22} x={97} y={34} speed={0.35} reverse offset={9} className="accent" />
        <Gear teeth={8} r={18} x={93} y={82} speed={0.35} reverse offset={3} hub={false} />
      </svg>
      <div className="loader-steps">
        <span>ML suitability</span>
        <span>Cost layer</span>
        <span>TOPSIS ranking</span>
      </div>
      <span className="loader-label">{label}</span>
    </div>
  );
}

// ---------- logo: user image at /logo.png, gear drawing as fallback ----------
export function Logo({ size = 34 }) {
  const [failed, setFailed] = useState(false);
  if (!failed) {
    return <img src="/logo.png" alt="Sigma" width={size} height={size} className="logo-img" onError={() => setFailed(true)} />;
  }
  return (
    <svg viewBox="-50 -50 100 100" width={size} height={size} className="logo-svg" aria-label="Sigma">
      <path d={gearPath(12, 46, 37)} className="logo-gear" />
      <circle r="26" className="logo-inner" />
      <text x="0" y="10" textAnchor="middle" className="logo-m">σ</text>
    </svg>
  );
}

// ---------- rocket: flies above the text, bounces off the mechanical drawings ----------
const RSIZE = 56;          // rocket size in px
const RR = RSIZE * 0.3;    // collision radius

// obstacles in the parent's coordinates: circles (gears, crank, bearing) and boxes (cylinder)
function obstacles(box) {
  const o = box.getBoundingClientRect();
  const out = [];
  const circ = (el) => {
    const b = el.getBoundingClientRect();
    if (b.width) out.push({ c: true, x: b.left - o.left + b.width / 2, y: b.top - o.top + b.height / 2, r: Math.min(b.width, b.height) / 2 });
  };
  const rect = (el) => {
    const b = el.getBoundingClientRect();
    if (b.width) out.push({ c: false, l: b.left - o.left, t: b.top - o.top, r2: b.right - o.left, b2: b.bottom - o.top });
  };
  const root = box.parentElement;
  root.querySelectorAll(".hero-art .gear-body, .hero-art .crank-disc, .hero-art .race-outer").forEach(circ);
  root.querySelectorAll(".hero-art .cylinder").forEach(rect);
  return out;
}

export function WanderingRocket() {
  const ref = useRef(null);
  useEffect(() => {
    if (window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) return;
    const el = ref.current;
    const box = el?.parentElement;
    if (!el || !box) return;
    let x = box.clientWidth * 0.3, y = box.clientHeight * 0.9, heading = -Math.PI / 3, turn = 0;
    const speed = 90;
    let last = performance.now(), nextTurn = 0, nextScan = 0, obs = [], raf;

    const step = (t) => {
      const dt = Math.min(0.05, (t - last) / 1000);
      last = t;
      const W = box.clientWidth, H = box.clientHeight;
      if (t > nextScan) { obs = obstacles(box); nextScan = t + 300; }
      if (t > nextTurn) { turn = (Math.random() - 0.5) * 1.6; nextTurn = t + 1200 + Math.random() * 2200; }
      heading += turn * dt;
      let vx = Math.cos(heading) * speed, vy = Math.sin(heading) * speed;
      x += vx * dt;
      y += vy * dt;

      // page edges
      if (x < RR) { x = RR; vx = Math.abs(vx); }
      if (x > W - RR) { x = W - RR; vx = -Math.abs(vx); }
      if (y < RR) { y = RR; vy = Math.abs(vy); }
      if (y > H - RR) { y = H - RR; vy = -Math.abs(vy); }

      // drawings: push out and reflect the velocity about the contact normal
      for (const ob of obs) {
        let nx, ny, gap;
        if (ob.c) {
          nx = x - ob.x; ny = y - ob.y;
          const d = Math.hypot(nx, ny) || 1;
          gap = ob.r + RR - d;
          nx /= d; ny /= d;
        } else {
          const cx = Math.max(ob.l, Math.min(x, ob.r2)), cy = Math.max(ob.t, Math.min(y, ob.b2));
          nx = x - cx; ny = y - cy;
          let d = Math.hypot(nx, ny);
          if (d === 0) { nx = x < (ob.l + ob.r2) / 2 ? -1 : 1; ny = 0; d = 1; gap = RR; }
          else { gap = RR - d; nx /= d; ny /= d; }
        }
        if (gap > 0) {
          x += nx * gap; y += ny * gap;
          const dot = vx * nx + vy * ny;
          if (dot < 0) { vx -= 2 * dot * nx; vy -= 2 * dot * ny; turn = 0; }
        }
      }
      heading = Math.atan2(vy, vx);
      el.style.transform = `translate(${x - RSIZE / 2}px, ${y - RSIZE / 2}px) rotate(${heading + Math.PI / 2}rad)`;
      raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, []);

  return (
    <svg ref={ref} className="rocket" viewBox="-16 -16 32 32" width={RSIZE} height={RSIZE} aria-hidden="true">
      <g className="flame"><path d="M-3 9 L0 15 L3 9 Z" /></g>
      <path d="M-4.5 -4 L-9 4 L-9 8 L-4.5 6 Z" className="rocket-fin" />
      <path d="M4.5 -4 L9 4 L9 8 L4.5 6 Z" className="rocket-fin" />
      <path d="M0 -15 C5 -10 5.5 -4 4.5 3 L4.5 9 L-4.5 9 L-4.5 3 C-5.5 -4 -5 -10 0 -15 Z" className="rocket-body" />
      <circle cx="0" cy="-5" r="2.4" className="rocket-window" />
    </svg>
  );
}
