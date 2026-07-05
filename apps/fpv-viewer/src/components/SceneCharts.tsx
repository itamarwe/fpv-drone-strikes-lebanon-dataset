import { useEffect, useRef } from "react";
import type { SceneTimeline, TimelinePoint } from "../three/sceneViewer";

// Two stacked chart cards in the style of the full tool's plots:
//   "Speed vs. Time"        — teal raw + amber smoothed + dashed average
//   "Height Above Ground"   — green line with an amber playhead dot
// Both share the flight-time axis, show a live readout on the right, and
// support click/drag to seek.

const PAD_L = 44;
const PAD_R = 12;
const PAD_T = 10;
const PAD_B = 18;

const COLORS = {
  raw: "rgba(45, 106, 106, 0.9)",
  smooth: "#ffb000",
  avg: "rgba(255, 255, 255, 0.35)",
  height: "#7ee081",
  playhead: "#ffb000",
  grid: "rgba(255,255,255,0.08)",
  label: "#a1a1a1",
};

function valueAt(points: TimelinePoint[], t: number): TimelinePoint {
  let best = points[0];
  for (const p of points) {
    if (Math.abs(p.t - best.t) < 1e-9) best = p;
    if (Math.abs(p.t - t) < Math.abs(best.t - t)) best = p;
  }
  return best;
}

function useChart(
  draw: (ctx: CanvasRenderingContext2D, w: number, h: number) => void,
  deps: unknown[],
) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const render = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      if (!w || !h) return;
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, w, h);
      draw(ctx, w, h);
    };
    render();
    const ro = new ResizeObserver(render);
    ro.observe(canvas);
    return () => ro.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return ref;
}

function frame(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  t0: number,
  t1: number,
  yMax: number,
  yLabel: (v: number) => string,
) {
  const plotH = h - PAD_T - PAD_B;
  ctx.font = "10px Geist, sans-serif";
  ctx.textBaseline = "middle";
  for (let i = 0; i <= 3; i += 1) {
    const fy = PAD_T + (plotH * i) / 3;
    ctx.strokeStyle = COLORS.grid;
    ctx.beginPath();
    ctx.moveTo(PAD_L, fy);
    ctx.lineTo(w - PAD_R, fy);
    ctx.stroke();
    ctx.fillStyle = COLORS.label;
    ctx.textAlign = "right";
    ctx.fillText(yLabel(yMax * (1 - i / 3)), PAD_L - 6, fy);
  }
  ctx.fillStyle = COLORS.label;
  ctx.textAlign = "left";
  ctx.fillText("0.0s", PAD_L, h - 7);
  ctx.textAlign = "right";
  ctx.fillText(`${(t1 - t0).toFixed(1)}s`, w - PAD_R, h - 7);
}

function line(
  ctx: CanvasRenderingContext2D,
  points: TimelinePoint[],
  pick: (p: TimelinePoint) => number | null,
  x: (t: number) => number,
  y: (v: number) => number,
  color: string,
  width: number,
) {
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineJoin = "round";
  ctx.beginPath();
  let started = false;
  for (const p of points) {
    const v = pick(p);
    if (v === null) continue;
    if (!started) {
      ctx.moveTo(x(p.t), y(v));
      started = true;
    } else ctx.lineTo(x(p.t), y(v));
  }
  ctx.stroke();
}

function playhead(
  ctx: CanvasRenderingContext2D,
  h: number,
  px: number,
  dotY: number | null,
) {
  ctx.strokeStyle = COLORS.playhead;
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  ctx.moveTo(px, PAD_T - 3);
  ctx.lineTo(px, h - PAD_B + 3);
  ctx.stroke();
  if (dotY !== null) {
    ctx.fillStyle = COLORS.playhead;
    ctx.beginPath();
    ctx.arc(px, dotY, 4, 0, Math.PI * 2);
    ctx.fill();
  }
}

function SeekableCanvas({
  chartRef,
  t0,
  t1,
  onSeek,
}: {
  chartRef: React.RefObject<HTMLCanvasElement>;
  t0: number;
  t1: number;
  onSeek: (t: number) => void;
}) {
  const dragRef = useRef(false);
  const seekFromEvent = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const frac = (e.clientX - rect.left - PAD_L) / Math.max(rect.width - PAD_L - PAD_R, 1);
    onSeek(t0 + Math.min(1, Math.max(0, frac)) * (t1 - t0));
  };
  return (
    <canvas
      className="chart-canvas"
      ref={chartRef}
      onPointerDown={(e) => {
        dragRef.current = true;
        e.currentTarget.setPointerCapture(e.pointerId);
        seekFromEvent(e);
      }}
      onPointerMove={(e) => {
        if (dragRef.current) seekFromEvent(e);
      }}
      onPointerUp={() => {
        dragRef.current = false;
      }}
    />
  );
}

export function SceneCharts({
  timeline,
  currentT,
  onSeek,
}: {
  timeline: SceneTimeline;
  currentT: number;
  onSeek: (t: number) => void;
}) {
  const { t0, t1, points, avgSpeedMs } = timeline;
  const span = Math.max(t1 - t0, 1e-6);
  const clampedT = Math.min(Math.max(currentT, t0), t1);
  const cur = valueAt(points, clampedT);

  const speeds = points.map((p) => p.speedRawMs).filter((v): v is number => v !== null);
  const sMax = Math.max(1, ...speeds) * 1.12;
  const heights = points.map((p) => p.heightM).filter((v): v is number => v !== null);
  const hMax = Math.max(1, ...heights) * 1.12;

  const speedRef = useChart(
    (ctx, w, h) => {
      const plotW = w - PAD_L - PAD_R;
      const plotH = h - PAD_T - PAD_B;
      const x = (t: number) => PAD_L + ((t - t0) / span) * plotW;
      const y = (v: number) => PAD_T + plotH - (v / sMax) * plotH;
      frame(ctx, w, h, t0, t1, sMax, (v) => v.toFixed(0));
      if (avgSpeedMs !== null) {
        ctx.strokeStyle = COLORS.avg;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(PAD_L, y(avgSpeedMs));
        ctx.lineTo(w - PAD_R, y(avgSpeedMs));
        ctx.stroke();
        ctx.setLineDash([]);
      }
      line(ctx, points, (p) => p.speedRawMs, x, y, COLORS.raw, 1);
      line(ctx, points, (p) => p.speedMs, x, y, COLORS.smooth, 2);
      playhead(ctx, h, x(clampedT), cur.speedMs !== null ? y(cur.speedMs) : null);
    },
    [timeline, clampedT],
  );

  const heightRef = useChart(
    (ctx, w, h) => {
      const plotW = w - PAD_L - PAD_R;
      const plotH = h - PAD_T - PAD_B;
      const x = (t: number) => PAD_L + ((t - t0) / span) * plotW;
      const y = (v: number) => PAD_T + plotH - (v / hMax) * plotH;
      frame(ctx, w, h, t0, t1, hMax, (v) => v.toFixed(0));
      line(ctx, points, (p) => p.heightM, x, y, COLORS.height, 2);
      playhead(ctx, h, x(clampedT), cur.heightM !== null ? y(cur.heightM) : null);
    },
    [timeline, clampedT],
  );

  return (
    <div className="scene-charts">
      <div className="chart-card">
        <header>
          <span>Speed vs. Time</span>
          <span className="chart-readout">
            {cur.speedMs !== null ? `${cur.speedMs.toFixed(1)} m/s` : "--"}
            {avgSpeedMs !== null ? ` · avg ${avgSpeedMs.toFixed(1)}` : ""}
          </span>
        </header>
        <SeekableCanvas chartRef={speedRef} t0={t0} t1={t1} onSeek={onSeek} />
      </div>
      <div className="chart-card">
        <header>
          <span>Height Above Ground</span>
          <span className="chart-readout">
            {cur.heightM !== null ? `${cur.heightM.toFixed(1)}m AGL` : "--"}
            {cur.distM !== null ? ` | ${cur.distM.toFixed(0)}m tgt` : ""}
          </span>
        </header>
        <SeekableCanvas chartRef={heightRef} t0={t0} t1={t1} onSeek={onSeek} />
      </div>
    </div>
  );
}
