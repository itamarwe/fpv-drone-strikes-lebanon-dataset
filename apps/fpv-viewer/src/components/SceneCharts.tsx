import { useEffect, useRef } from "react";
import type { SceneTimeline } from "../three/sceneViewer";

// Dual-axis canvas chart: height above ground (m, left axis, blue) and flight
// speed (m/s, right axis, grey) against source-video time, with a synced
// playhead. Click or drag to seek.
export function SceneCharts({
  timeline,
  currentT,
  onSeek,
}: {
  timeline: SceneTimeline;
  currentT: number;
  onSeek: (t: number) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const dragRef = useRef(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const draw = () => {
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

      const padL = 40;
      const padR = 46;
      const padT = 16;
      const padB = 20;
      const plotW = w - padL - padR;
      const plotH = h - padT - padB;
      const { t0, t1, points } = timeline;
      const span = Math.max(t1 - t0, 1e-6);
      const x = (t: number) => padL + ((t - t0) / span) * plotW;

      const heights = points.map((p) => p.heightM).filter((v): v is number => v !== null);
      const speeds = points.map((p) => p.speedMs).filter((v): v is number => v !== null);
      const hMax = Math.max(1, ...heights) * 1.1;
      const sMax = Math.max(1, ...speeds) * 1.1;
      const yH = (v: number) => padT + plotH - (v / hMax) * plotH;
      const yS = (v: number) => padT + plotH - (v / sMax) * plotH;

      // horizontal grid lines + axis labels
      ctx.font = "10px Geist, sans-serif";
      ctx.textBaseline = "middle";
      for (let i = 0; i <= 3; i += 1) {
        const fy = padT + (plotH * i) / 3;
        ctx.strokeStyle = "rgba(255,255,255,0.08)";
        ctx.beginPath();
        ctx.moveTo(padL, fy);
        ctx.lineTo(w - padR, fy);
        ctx.stroke();
        const hVal = hMax * (1 - i / 3);
        const sVal = sMax * (1 - i / 3);
        ctx.fillStyle = "#3291ff";
        ctx.textAlign = "right";
        ctx.fillText(hVal.toFixed(0), padL - 6, fy);
        ctx.fillStyle = "#a1a1a1";
        ctx.textAlign = "left";
        ctx.fillText(sVal.toFixed(0), w - padR + 6, fy);
      }
      ctx.fillStyle = "#3291ff";
      ctx.textAlign = "left";
      ctx.fillText("height m", padL, 8);
      ctx.fillStyle = "#a1a1a1";
      ctx.textAlign = "right";
      ctx.fillText("speed m/s", w - padR, 8);

      // time labels
      ctx.fillStyle = "#777";
      ctx.textAlign = "center";
      ctx.fillText(`${t0.toFixed(1)}s`, padL, h - 8);
      ctx.fillText(`${t1.toFixed(1)}s`, w - padR, h - 8);

      const series = (
        pick: (p: (typeof points)[number]) => number | null,
        yFn: (v: number) => number,
        color: string,
      ) => {
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        let started = false;
        for (const p of points) {
          const v = pick(p);
          if (v === null) continue;
          const px = x(p.t);
          const py = yFn(v);
          if (!started) {
            ctx.moveTo(px, py);
            started = true;
          } else ctx.lineTo(px, py);
        }
        ctx.stroke();
      };
      series((p) => p.heightM, yH, "#3291ff");
      series((p) => p.speedMs, yS, "#a1a1a1");

      // playhead
      const px = x(Math.min(Math.max(currentT, t0), t1));
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.moveTo(px, padT - 4);
      ctx.lineTo(px, padT + plotH + 4);
      ctx.stroke();
    };
    draw();
    const ro = new ResizeObserver(draw);
    ro.observe(canvas);
    return () => ro.disconnect();
  }, [timeline, currentT]);

  const seekFromEvent = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const padL = 40;
    const padR = 46;
    const frac = (e.clientX - rect.left - padL) / Math.max(rect.width - padL - padR, 1);
    const t = timeline.t0 + Math.min(1, Math.max(0, frac)) * (timeline.t1 - timeline.t0);
    onSeek(t);
  };

  return (
    <canvas
      className="scene-chart"
      ref={canvasRef}
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
