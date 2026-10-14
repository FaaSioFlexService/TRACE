/**
 * Laboratorio visual de benchmarking.
 *
 * La aplicación está diseñada como herramienta de ingeniería: primero configura
 * un banco reproducible, después lo ejecuta y finalmente analiza el historial.
 */

import { useMemo, useState } from "react";
import {
  generateBenchmarkCases,
  runBenchmarkBank,
  type BenchmarkCase,
} from "./benchmark";
import type {
  BenchmarkConfiguration,
  BenchmarkSample,
  MatrixFormat,
  OperationName,
  ScalarType,
} from "@trace/vector-matrix-core";

const SIZE_OPTIONS = [64, 128, 256, 512, 1024, 2048];

const OPERATION_LABELS: Record<OperationName, string> = {
  vectorAdd: "Vector addition",
  vectorScale: "Vector scaling",
  dotProduct: "Dot product",
  vectorNorm2: "Vector norm",
  matrixVectorMultiply: "Matrix × vector",
  matrixMatrixMultiply: "Matrix × matrix",
};

/**
 * Formatea bytes para la tabla.
 */
function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes.toFixed(0)} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
}

/**
 * Formatea un tiempo en una unidad útil para lectura humana.
 */
function formatMilliseconds(milliseconds: number): string {
  if (milliseconds < 0.01) return `${(milliseconds * 1000).toFixed(2)} µs`;
  if (milliseconds < 1) return `${milliseconds.toFixed(3)} ms`;
  return `${milliseconds.toFixed(2)} ms`;
}

/**
 * Descarga texto generado desde el navegador.
 */
function downloadText(filename: string, content: string, type: string): void {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

/**
 * Convierte muestras a CSV para análisis externo.
 */
function samplesToCsv(samples: BenchmarkSample[]): string {
  const headers = [
    "timestamp",
    "operation",
    "matrixFormat",
    "scalarType",
    "rows",
    "columns",
    "vectorLength",
    "density",
    "warmupIterations",
    "measuredIterations",
    "medianMilliseconds",
    "meanMilliseconds",
    "minMilliseconds",
    "maxMilliseconds",
    "standardDeviationMilliseconds",
    "estimatedGflops",
    "estimatedMemoryBytes",
    "backend",
  ];

  const rows = samples.map((sample) =>
    headers.map((header) => JSON.stringify(sample[header as keyof BenchmarkSample])).join(","),
  );

  return [headers.join(","), ...rows].join("\n");
}

/**
 * Calcula un resumen del historial.
 */
function summarize(samples: BenchmarkSample[]) {
  if (samples.length === 0) {
    return {
      runs: 0,
      median: 0,
      bestGflops: 0,
      totalMemory: 0,
    };
  }

  const medians = samples.map((sample) => sample.medianMilliseconds).sort((a, b) => a - b);
  return {
    runs: samples.length,
    median: medians[Math.floor(medians.length / 2)],
    bestGflops: Math.max(...samples.map((sample) => sample.estimatedGflops)),
    totalMemory: Math.max(...samples.map((sample) => sample.estimatedMemoryBytes)),
  };
}

/**
 * Renderiza la aplicación completa.
 */
export default function App() {
  const [operation, setOperation] = useState<OperationName>("matrixVectorMultiply");
  const [formats, setFormats] = useState<MatrixFormat[]>(["dense", "csr"]);
  const [scalarTypes, setScalarTypes] = useState<ScalarType[]>(["float64"]);
  const [sizes, setSizes] = useState<number[]>([64, 128, 256, 512]);
  const [densities, setDensities] = useState<number[]>([1, 0.1, 0.01]);
  const [warmupIterations, setWarmupIterations] = useState(3);
  const [measuredIterations, setMeasuredIterations] = useState(15);
  const [casesPerConfiguration, setCasesPerConfiguration] = useState(1);
  const [history, setHistory] = useState<BenchmarkSample[]>([]);
  const [progress, setProgress] = useState({ completed: 0, total: 0 });
  const [running, setRunning] = useState(false);

  const configuration: BenchmarkConfiguration = useMemo(
    () => ({
      operation,
      matrixFormats: formats,
      scalarTypes,
      sizes,
      densities,
      warmupIterations,
      measuredIterations,
      casesPerConfiguration,
      backend: "typescript-baseline",
    }),
    [
      operation,
      formats,
      scalarTypes,
      sizes,
      densities,
      warmupIterations,
      measuredIterations,
      casesPerConfiguration,
    ],
  );

  const summary = summarize(history);
  const previewCases: BenchmarkCase[] = generateBenchmarkCases(configuration).slice(0, 12);

  /**
   * Alterna una opción dentro de una lista de selección.
   */
  function toggleValue<T>(values: T[], value: T, setter: (next: T[]) => void): void {
    setter(
      values.includes(value)
        ? values.filter((current) => current !== value)
        : [...values, value],
    );
  }

  /**
   * Ejecuta el banco actual y acumula resultados en el historial.
   */
  async function handleRun(): Promise<void> {
    setRunning(true);
    setProgress({ completed: 0, total: generateBenchmarkCases(configuration).length });

    try {
      const results = await runBenchmarkBank(configuration, (completed, total) => {
        setProgress({ completed, total });
      });
      setHistory((current) => [...results, ...current]);
    } finally {
      setRunning(false);
    }
  }

  /**
   * Borra el historial para iniciar una campaña independiente.
   */
  function clearHistory(): void {
    setHistory([]);
    setProgress({ completed: 0, total: 0 });
  }

  const report = {
    generatedAt: new Date().toISOString(),
    configuration,
    summary,
    results: history,
  };

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <div className="eyebrow">TRACE / HPC LABORATORY</div>
          <h1>Vector & Matrix Benchmark Lab</h1>
          <p>
            Reproducible performance laboratory for the independent numerical
            core that will later power TRACE.
          </p>
        </div>
        <div className="status-badge">
          <span className="status-dot" />
          CPU baseline
        </div>
      </header>

      <section className="metrics-grid">
        <article className="metric-card">
          <span>Recorded runs</span>
          <strong>{summary.runs}</strong>
        </article>
        <article className="metric-card">
          <span>Median runtime</span>
          <strong>{summary.runs ? formatMilliseconds(summary.median) : "—"}</strong>
        </article>
        <article className="metric-card">
          <span>Peak estimated GFLOP/s</span>
          <strong>{summary.runs ? summary.bestGflops.toFixed(3) : "—"}</strong>
        </article>
        <article className="metric-card">
          <span>Largest working set</span>
          <strong>{summary.runs ? formatBytes(summary.totalMemory) : "—"}</strong>
        </article>
      </section>

      <div className="workspace">
        <aside className="panel controls">
          <div className="panel-heading">
            <div>
              <span className="section-kicker">01 / DESIGN</span>
              <h2>Test bank</h2>
            </div>
            <span className="pill">Baseline</span>
          </div>

          <label>
            Operation
            <select
              value={operation}
              onChange={(event) => setOperation(event.target.value as OperationName)}
            >
              {Object.entries(OPERATION_LABELS).map(([value, label]) => (
                <option key={value} value={value}>{label}</option>
              ))}
            </select>
          </label>

          {operation.startsWith("matrix") && (
            <fieldset>
              <legend>Matrix formats</legend>
              <div className="check-grid">
                {(["dense", "csr"] as MatrixFormat[]).map((format) => (
                  <label className="check-row" key={format}>
                    <input
                      type="checkbox"
                      checked={formats.includes(format)}
                      onChange={() => toggleValue(formats, format, setFormats)}
                    />
                    {format === "dense" ? "Dense / row-major" : "Sparse / CSR"}
                  </label>
                ))}
              </div>
            </fieldset>
          )}

          <fieldset>
            <legend>Precision</legend>
            <div className="check-grid">
              {(["float32", "float64"] as ScalarType[]).map((type) => (
                <label className="check-row" key={type}>
                  <input
                    type="checkbox"
                    checked={scalarTypes.includes(type)}
                    onChange={() => toggleValue(scalarTypes, type, setScalarTypes)}
                  />
                  {type.toUpperCase()}
                </label>
              ))}
            </div>
          </fieldset>

          <fieldset>
            <legend>Matrix / vector sizes</legend>
            <div className="check-grid sizes">
              {SIZE_OPTIONS.map((size) => (
                <label className="check-row" key={size}>
                  <input
                    type="checkbox"
                    checked={sizes.includes(size)}
                    onChange={() => toggleValue(sizes, size, setSizes)}
                  />
                  {size}×{size}
                </label>
              ))}
            </div>
          </fieldset>

          {operation.startsWith("matrix") && (
            <fieldset>
              <legend>Density</legend>
              <div className="check-grid">
                {[1, 0.1, 0.01].map((density) => (
                  <label className="check-row" key={density}>
                    <input
                      type="checkbox"
                      checked={densities.includes(density)}
                      onChange={() => toggleValue(densities, density, setDensities)}
                    />
                    {density === 1 ? "100%" : `${density * 100}%`}
                  </label>
                ))}
              </div>
            </fieldset>
          )}

          <div className="two-column">
            <label>
              Warmup
              <input
                type="number"
                min="0"
                max="50"
                value={warmupIterations}
                onChange={(event) => setWarmupIterations(Number(event.target.value))}
              />
            </label>
            <label>
              Samples
              <input
                type="number"
                min="3"
                max="100"
                value={measuredIterations}
                onChange={(event) => setMeasuredIterations(Number(event.target.value))}
              />
            </label>
          </div>

          <label>
            Repetitions / configuration
            <input
              type="number"
              min="1"
              max="10"
              value={casesPerConfiguration}
              onChange={(event) => setCasesPerConfiguration(Number(event.target.value))}
            />
          </label>

          <button
            className="primary-button"
            disabled={running || sizes.length === 0 || scalarTypes.length === 0}
            onClick={handleRun}
          >
            {running ? "Running benchmark…" : "Generate & run test bank"}
          </button>

          <div className="progress">
            <div className="progress-label">
              <span>{progress.completed} / {progress.total} cases</span>
              <span>
                {progress.total
                  ? `${Math.round((progress.completed / progress.total) * 100)}%`
                  : "Ready"}
              </span>
            </div>
            <div className="progress-track">
              <div
                className="progress-value"
                style={{
                  width: `${progress.total ? (progress.completed / progress.total) * 100 : 0}%`,
                }}
              />
            </div>
          </div>
        </aside>

        <section className="content-stack">
          <article className="panel">
            <div className="panel-heading">
              <div>
                <span className="section-kicker">02 / PLAN</span>
                <h2>Generated cases</h2>
              </div>
              <span className="pill">{generateBenchmarkCases(configuration).length} total</span>
            </div>

            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Case</th>
                    <th>Operation</th>
                    <th>Format</th>
                    <th>Precision</th>
                    <th>Size</th>
                    <th>Density</th>
                  </tr>
                </thead>
                <tbody>
                  {previewCases.map((testCase) => (
                    <tr key={testCase.id}>
                      <td className="mono">{testCase.id}</td>
                      <td>{OPERATION_LABELS[testCase.operation]}</td>
                      <td>{testCase.matrixFormat}</td>
                      <td>{testCase.scalarType}</td>
                      <td>{testCase.size}</td>
                      <td>{testCase.density * 100}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </article>

          <article className="panel">
            <div className="panel-heading">
              <div>
                <span className="section-kicker">03 / RESULTS</span>
                <h2>Performance history</h2>
              </div>
              <div className="button-group">
                <button
                  className="secondary-button"
                  disabled={!history.length}
                  onClick={() =>
                    downloadText(
                      "trace-benchmark.json",
                      JSON.stringify(report, null, 2),
                      "application/json",
                    )
                  }
                >
                  Export JSON
                </button>
                <button
                  className="secondary-button"
                  disabled={!history.length}
                  onClick={() =>
                    downloadText("trace-benchmark.csv", samplesToCsv(history), "text/csv")
                  }
                >
                  Export CSV
                </button>
                <button className="danger-button" disabled={!history.length} onClick={clearHistory}>
                  Clear
                </button>
              </div>
            </div>

            <div className="table-wrap history-table">
              <table>
                <thead>
                  <tr>
                    <th>Operation</th>
                    <th>Format</th>
                    <th>Type</th>
                    <th>Size</th>
                    <th>Density</th>
                    <th>Median</th>
                    <th>GFLOP/s</th>
                    <th>Memory</th>
                  </tr>
                </thead>
                <tbody>
                  {history.slice(0, 50).map((sample) => (
                    <tr key={`${sample.id}-${sample.timestamp}`}>
                      <td>{OPERATION_LABELS[sample.operation]}</td>
                      <td>{sample.matrixFormat}</td>
                      <td>{sample.scalarType}</td>
                      <td>{sample.rows}×{sample.columns}</td>
                      <td>{sample.density * 100}%</td>
                      <td className="mono">{formatMilliseconds(sample.medianMilliseconds)}</td>
                      <td className="mono">{sample.estimatedGflops.toFixed(3)}</td>
                      <td>{formatBytes(sample.estimatedMemoryBytes)}</td>
                    </tr>
                  ))}
                  {!history.length && (
                    <tr>
                      <td colSpan={8} className="empty-state">
                        No measurements yet. Configure a bank and run it.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </article>

          <article className="panel report-panel">
            <div>
              <span className="section-kicker">04 / ENGINEERING NOTES</span>
              <h2>Benchmark methodology</h2>
            </div>
            <div className="method-grid">
              <div>
                <strong>Warmup</strong>
                <p>Separates initialization and JIT effects from measured samples.</p>
              </div>
              <div>
                <strong>Median</strong>
                <p>Primary latency metric; less sensitive to occasional browser noise.</p>
              </div>
              <div>
                <strong>Density</strong>
                <p>Required to identify the crossover point where sparse storage becomes beneficial.</p>
              </div>
              <div>
                <strong>Baseline first</strong>
                <p>Future WASM/SIMD/thread backends will be compared against this exact API.</p>
              </div>
            </div>
          </article>
        </section>
      </div>
    </main>
  );
}
