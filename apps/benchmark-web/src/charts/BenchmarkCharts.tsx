/**
 * Capa de visualización del laboratorio.
 *
 * Los gráficos muestran métricas comparables y conservan una identidad de
 * color estable por backend. No realizan cálculos de benchmark ni modifican
 * resultados persistidos.
 */

import type { BenchmarkResult } from "../benchmark";
import { useI18n } from "../i18n";

interface Point {
  readonly x: number;
  readonly y: number;
  readonly label: string;
}

export interface Series {
  readonly name: string;
  readonly points: Point[];
  readonly color: string;
  readonly isOwn?: boolean;
}

interface LineChartProps {
  readonly title: string;
  readonly yLabel: string;
  readonly series: Series[];
  readonly formatY?: (value: number) => string;
}

/**
 * Dibuja una curva comparable sobre el eje de tamaño del problema.
 */
export function LineChart({ title, yLabel, series, formatY = (value) => value.toFixed(2) }: LineChartProps) {
  const { t } = useI18n();
  const width = 760;
  const height = 300;
  const padding = { left: 64, right: 24, top: 30, bottom: 48 };
  const points = series.flatMap((item) => item.points);
  const maxX = Math.max(...points.map((point) => point.x), 1);
  const maxY = Math.max(...points.map((point) => point.y), 1);

  function xScale(value: number): number {
    return padding.left + (value / maxX) * (width - padding.left - padding.right);
  }

  function yScale(value: number): number {
    return height - padding.bottom - (value / maxY) * (height - padding.top - padding.bottom);
  }

  return (
    <div className="chart-card">
      <div className="chart-heading"><strong>{title}</strong><span>{yLabel}</span></div>
      {series.length === 0 ? <div className="chart-empty">{t("No compatible results for this selection.")}</div> : (
        <>
          <svg className="chart-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={title}>
            {[0, 0.25, 0.5, 0.75, 1].map((tick) => {
              const value = maxY * tick;
              const y = yScale(value);
              return <g key={tick}><line x1={padding.left} x2={width - padding.right} y1={y} y2={y} className="chart-grid" /><text x={padding.left - 8} y={y + 4} textAnchor="end" className="chart-axis-label"><title>{`${t("Chart axis value")}: ${formatY(value)}`}</title>{formatY(value)}</text></g>;
            })}
            {series.map((item) => {
              const path = item.points.map((point, index) => `${index === 0 ? "M" : "L"}${xScale(point.x)} ${yScale(point.y)}`).join(" ");
              return <g key={item.name}><path d={path} className="chart-line" stroke={item.color} />{item.points.map((point) => <circle key={`${item.name}-${point.label}`} cx={xScale(point.x)} cy={yScale(point.y)} r="4" className="chart-point" fill={item.color}><title>{`${item.name} · ${t("Problem size")}: ${point.label} · ${yLabel}: ${formatY(point.y)}`}</title></circle>)}</g>;
            })}
          </svg>
          <div className="chart-legend">{series.map((item) => <span className={item.isOwn ? "legend-item own" : "legend-item"} key={item.name}><i className="legend-dot" style={{ background: item.color }} />{item.name}{item.isOwn ? ` · ${t("TRACE · OUR LIBRARY")}` : ""}</span>)}</div>
        </>
      )}
    </div>
  );
}

interface BarChartProps {
  readonly title: string;
  readonly values: readonly { label: string; value: number; color: string; isOwn?: boolean }[];
  readonly formatValue?: (value: number) => string;
}

/**
 * Dibuja barras horizontales para ratios relativos a TRACE.
 */
export function BarChart({ title, values, formatValue = (value) => `${value.toFixed(2)}×` }: BarChartProps) {
  const { t } = useI18n();
  const maxValue = Math.max(...values.map((item) => item.value), 1);
  return (
    <div className="chart-card">
      <div className="chart-heading"><strong>{title}</strong><span>{t("relative to TRACE")}</span></div>
      {values.length === 0 ? <div className="chart-empty">{t("Select at least one library executed in this campaign.")}</div> : (
        <div className="bar-chart">{values.map((item) => <div className="bar-row" key={item.label}><span className={item.isOwn ? "bar-label own" : "bar-label"}>{item.label}</span><div className="bar-track"><div className="bar-value" style={{ width: `${(item.value / maxValue) * 100}%`, background: item.color }} /></div><strong>{formatValue(item.value)}</strong></div>)}</div>
      )}
    </div>
  );
}

interface ErrorBarChartProps {
  readonly title: string;
  readonly series: Series[];
  readonly intervals: Map<string, { low: number; high: number }>;
  readonly formatY?: (value: number) => string;
}

/**
 * Representa P50 junto al intervalo de confianza aproximado almacenado.
 */
export function ErrorBarChart({ title, series, intervals, formatY = (value) => value.toFixed(2) }: ErrorBarChartProps) {
  const { t } = useI18n();
  const width = 760;
  const height = 300;
  const padding = { left: 64, right: 24, top: 30, bottom: 48 };
  const points = series.flatMap((item) => item.points);
  const maxX = Math.max(...points.map((point) => point.x), 1);
  const maxY = Math.max(...[...intervals.values()].map((interval) => interval.high), 1);
  const xScale = (value: number) => padding.left + (value / maxX) * (width - padding.left - padding.right);
  const yScale = (value: number) => height - padding.bottom - (value / maxY) * (height - padding.top - padding.bottom);

  return (
    <div className="chart-card">
      <div className="chart-heading"><strong>{title}</strong><span>{t("P50 + 95% CI")}</span></div>
      {series.length === 0 ? <div className="chart-empty">{t("No compatible results.")}</div> : (
        <>
          <svg className="chart-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={title}>
            {[0, 0.25, 0.5, 0.75, 1].map((tick) => { const value = maxY * tick; const y = yScale(value); return <g key={tick}><line x1={padding.left} x2={width - padding.right} y1={y} y2={y} className="chart-grid" /><text x={padding.left - 8} y={y + 4} textAnchor="end" className="chart-axis-label"><title>{`${t("Chart axis value")}: ${formatY(value)}`}</title>{formatY(value)}</text></g>; })}
            {series.map((item) => item.points.map((point) => { const interval = intervals.get(`${item.name}|${point.label}`); if (!interval) return null; const x = xScale(point.x); return <g key={`${item.name}-${point.label}`}><line x1={x} x2={x} y1={yScale(interval.low)} y2={yScale(interval.high)} stroke={item.color} strokeWidth="3" /><line x1={x - 5} x2={x + 5} y1={yScale(interval.low)} y2={yScale(interval.low)} stroke={item.color} /><line x1={x - 5} x2={x + 5} y1={yScale(interval.high)} y2={yScale(interval.high)} stroke={item.color} /><circle cx={x} cy={yScale(point.y)} r="5" fill={item.color} className="chart-point"><title>{`${item.name} · ${t("P50 explanation")} · ${formatY(point.y)}`}</title></circle></g>; }))}
          </svg>
          <div className="chart-legend">{series.map((item) => <span className={item.isOwn ? "legend-item own" : "legend-item"} key={item.name}><i className="legend-dot" style={{ background: item.color }} />{item.name}</span>)}</div>
        </>
      )}
    </div>
  );
}

/**
 * Agrupa resultados por backend y tamaño para cualquier métrica profesional.
 */
export function buildSeries(
  results: BenchmarkResult[],
  metric: "runtime" | "gflops" | "memory" | "cv" | "tailRatio",
  colorResolver: (backendId: BenchmarkResult["backendId"]) => string,
): Series[] {
  const grouped = new Map<string, { backendId: BenchmarkResult["backendId"]; bySize: Map<number, BenchmarkResult> }>();
  for (const result of results) {
    if (result.status !== "completed") continue;
    if (!grouped.has(result.backendName)) grouped.set(result.backendName, { backendId: result.backendId, bySize: new Map() });
    grouped.get(result.backendName)!.bySize.set(result.rows, result);
  }

  return [...grouped.entries()].map(([name, group]) => ({
    name,
    color: colorResolver(group.backendId),
    isOwn: group.backendId === "trace-core",
    points: [...group.bySize.entries()].sort(([left], [right]) => left - right).map(([size, result]) => ({
      x: size,
      y: metric === "runtime" ? result.statistics.p50
        : metric === "gflops" ? result.estimatedGflops
          : metric === "memory" ? result.estimatedMemoryBytes
            : metric === "cv" ? result.statistics.coefficientOfVariation * 100
              : result.statistics.p50 > 0 ? result.statistics.p95 / result.statistics.p50 : 0,
      label: String(size),
    })),
  })).filter((series) => series.points.length > 0);
}
