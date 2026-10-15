/**
 * Aplicación principal del TRACE HPC Benchmark Lab.
 *
 * El shell ocupa todo el viewport, la configuración queda siempre visible y
 * cada pestaña presenta una vista compacta sin scroll de página. Las tablas se
 * limitan mediante paginación externa para mantener una lectura profesional.
 */

import { useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import {
  BENCHMARK_BACKENDS,
  NATIVE_BENCHMARK_BACKENDS,
  generateBenchmarkCases,
  runServerBenchmarkBank,
  readEnvironmentSnapshot,
  type BenchmarkBackendId,
  type BenchmarkConfiguration,
  type BenchmarkResult,
  type BenchmarkRun,
  type OperationName,
  type BenchmarkSnapshot,
  type UploadedBenchmarkAdapterDefinition,
} from "./benchmark";
import {
  clearBenchmarkHistory,
  deleteBenchmarkResults,
  loadBenchmarkHistory,
  saveBenchmarkSnapshot,
} from "./history";
import {
  AdapterUpload,
  BackendSelector,
  DataTable,
  MetricCard,
  Pagination,
  Tabs,
  ThemeToggle,
  LanguageToggle,
  ExecutionWorldSelector,
  InfoTooltip,
  NumericValue,
  type TableColumn,
} from "./components";
import { BarChart, ErrorBarChart, LineChart, buildSeries } from "./charts/BenchmarkCharts";
import { OPERATION_CATALOG, type MatrixFormat, type ScalarType, type OperationCategory } from "@trace/vector-matrix-core";
import { useI18n } from "./i18n";
import type { ExecutionWorld, ServerExecutionPolicy } from "./components/ExecutionWorldSelector";

const SIZE_OPTIONS = [64, 128, 256, 512, 1024, 2048];
const TABLE_PAGE_SIZE = 8;
const HISTORY_PAGE_SIZE = 8;

const OPERATION_LABELS: Record<OperationName, string> = Object.fromEntries(
  OPERATION_CATALOG.map((spec) => [spec.id, spec.description]),
) as Record<OperationName, string>;

const BACKEND_COLORS: Record<string, string> = {
  "trace-core": "#2563eb",
  "javascript-baseline": "#f59e0b",
  "ml-matrix": "#a855f7",
  "ndarray-blas-gemv": "#14b8a6",
};

const CUSTOM_BACKEND_COLORS = ["#8b5cf6", "#ec4899", "#f97316", "#22c55e", "#06b6d4", "#eab308"];

/**
 * Devuelve un color estable para cualquier backend, incluido un adapter subido.
 */
function getBackendColor(backendId: string): string {
  if (BACKEND_COLORS[backendId]) return BACKEND_COLORS[backendId];
  let hash = 0;
  for (const character of backendId) hash = (hash * 31 + character.charCodeAt(0)) >>> 0;
  return CUSTOM_BACKEND_COLORS[hash % CUSTOM_BACKEND_COLORS.length];
}

type MainTab = "benchmark" | "comparison" | "charts" | "history" | "methodology" | "environment";
type Theme = "dark" | "light";
type ChartView = "overview" | "stability" | "memory" | "statistics";

const OPERATION_CATEGORY_LABELS: Record<OperationCategory, string> = {
  level1: "BLAS Level 1 · Vector",
  level2: "BLAS Level 2 · Matrix–Vector",
  level3: "BLAS Level 3 · Matrix–Matrix",
  matrix: "Matrix operations",
  factorization: "Matrix factorizations",
  eigen: "Eigen / spectral",
  exact: "Integer / polynomial",
};

const UPLOADED_ADAPTER_STORAGE_KEY = "trace-benchmark-uploaded-adapters-v1";

interface ComparisonRow {
  readonly id: string;
  readonly caseId: string;
  readonly operation: string;
  readonly size: string;
  readonly format: string;
  readonly density: string;
  readonly traceP50: number;
  readonly backendId: BenchmarkBackendId;
  readonly backendName: string;
  readonly competitorP50: number;
  readonly relative: number;
  readonly traceP50Explanation: string;
  readonly competitorP50Explanation: string;
  readonly relativeExplanation: string;
}

interface ChartSliceOption {
  readonly key: string;
  readonly operation: OperationName;
  readonly matrixFormat: MatrixFormat | "none";
  readonly scalarType: ScalarType;
  readonly densityRequested: number;
}

/**
 * Formatea bytes para lectura humana.
 */
function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes.toFixed(0)} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
}

/**
 * Formatea tiempos con una unidad legible sin ocultar diferencias pequeñas.
 */
function formatMilliseconds(milliseconds: number): string {
  if (milliseconds < 0.001) return `${(milliseconds * 1000).toFixed(3)} µs`;
  if (milliseconds < 1) return `${milliseconds.toFixed(3)} ms`;
  return `${milliseconds.toFixed(2)} ms`;
}

/**
 * Formatea ratios relativos a TRACE.
 */
function formatRatio(value: number): string {
  return `${value.toFixed(2)}×`;
}

/**
 * Descarga texto generado por el laboratorio.
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
 * Alterna una opción dentro de una selección.
 */
function toggleValue<T>(values: T[], value: T, setter: (next: T[]) => void): void {
  setter(values.includes(value) ? values.filter((current) => current !== value) : [...values, value]);
}

/**
 * Calcula la media de una métrica sobre resultados completados.
 */
function average(results: BenchmarkResult[], selector: (result: BenchmarkResult) => number): number {
  const values = results.filter((result) => result.status === "completed").map(selector);
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
}

/**
 * Obtiene una etiqueta corta para una campaña histórica.
 */
function runLabel(run: BenchmarkRun): string {
  return `${run.configuration.operation} · ${run.configuration.sizes.join("/")}`;
}

/**
 * Recupera adapters registrados en sesiones anteriores sin ejecutar su código.
 */
function loadUploadedAdapters(): UploadedBenchmarkAdapterDefinition[] {
  try {
    const raw = localStorage.getItem(UPLOADED_ADAPTER_STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as UploadedBenchmarkAdapterDefinition[];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

/**
 * Componente raíz del laboratorio.
 */
export default function App() {
  const { t } = useI18n();
  const [activeTab, setActiveTab] = useState<MainTab>("benchmark");
  const [executionWorld, setExecutionWorld] = useState<ExecutionWorld>(() => (localStorage.getItem("trace-benchmark-world") as ExecutionWorld) || "browser");
  const [serverUrl, setServerUrl] = useState(() => localStorage.getItem("trace-benchmark-server-url") || "http://localhost:8080");
  const [serverExecutionPolicy, setServerExecutionPolicy] = useState<ServerExecutionPolicy>(() => (localStorage.getItem("trace-benchmark-server-policy") as ServerExecutionPolicy) || "cpu");
  const [serverBackendAvailability, setServerBackendAvailability] = useState<Record<string, boolean>>({});
  const [serverConnected, setServerConnected] = useState(false);
  const [wasmAvailable, setWasmAvailable] = useState(false);
  const [theme, setTheme] = useState<Theme>(() => (localStorage.getItem("trace-benchmark-theme") as Theme) || "dark");
  const [operation, setOperation] = useState<OperationName>("matrixVectorMultiply");
  const [operationCategory, setOperationCategory] = useState<OperationCategory>("level2");
  const [formats, setFormats] = useState<MatrixFormat[]>(["dense", "csr", "csc", "ellpack"]);
  const [scalarTypes, setScalarTypes] = useState<ScalarType[]>(["float64"]);
  const [sizes, setSizes] = useState<number[]>([64, 128, 256, 512]);
  const [densities, setDensities] = useState<number[]>([1, 0.1, 0.01]);
  const [warmupIterations, setWarmupIterations] = useState(3);
  const [measuredIterations, setMeasuredIterations] = useState(15);
  const [casesPerConfiguration, setCasesPerConfiguration] = useState(1);
  const [targetSampleMilliseconds, setTargetSampleMilliseconds] = useState(8);
  const [uploadedBackends, setUploadedBackends] = useState<UploadedBenchmarkAdapterDefinition[]>(() => loadUploadedAdapters());
  const [selectedBackends, setSelectedBackends] = useState<BenchmarkBackendId[]>(["trace-core", "javascript-baseline", "ml-matrix", "ndarray-blas-gemv"]);
  const [runs, setRuns] = useState<BenchmarkRun[]>([]);
  const [results, setResults] = useState<BenchmarkResult[]>([]);
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [comparisonSelections, setComparisonSelections] = useState<Record<string, BenchmarkBackendId[]>>({});
  const [progress, setProgress] = useState({ completed: 0, total: 0 });
  const [running, setRunning] = useState(false);
  const benchmarkWorkerRef = useRef<Worker | null>(null);
  const benchmarkCancelRef = useRef<(() => void) | null>(null);
  const [previewPage, setPreviewPage] = useState(1);
  const [ownPage, setOwnPage] = useState(1);
  const [comparisonPage, setComparisonPage] = useState(1);
  const [historyPage, setHistoryPage] = useState(1);
  const [historyBackendFilter, setHistoryBackendFilter] = useState("all");
  const [historyRunFilter, setHistoryRunFilter] = useState("all");
  const [historyOperationFilter, setHistoryOperationFilter] = useState("all");
  const [selectedResultIds, setSelectedResultIds] = useState<Set<string>>(new Set());
  const [chartSliceKey, setChartSliceKey] = useState("");
  const [chartView, setChartView] = useState<ChartView>("overview");

  const operationLabel = (value: OperationName): string => t(OPERATION_LABELS[value]);
  const selectedOperationSpec = OPERATION_CATALOG.find((spec) => spec.id === operation);
  const matrixOperation = selectedOperationSpec?.category !== "level1";
  const ownBackendIds = executionWorld === "server" ? ["trace-rust-native-cpu", "trace-rust-native-gpu"] : ["trace-rust-wasm", "trace-core"];
  const operationCategories = OPERATION_CATALOG.reduce<Record<OperationCategory, typeof OPERATION_CATALOG[number][]>>((groups, spec) => {
    (groups[spec.category] ??= []).push(spec);
    return groups;
  }, {} as Record<OperationCategory, typeof OPERATION_CATALOG[number][]>);

  const configuration: BenchmarkConfiguration = useMemo(() => ({
    operation,
    matrixFormats: operation === "matrixVectorMultiply" ? formats : matrixOperation ? ["dense"] : [],
    scalarTypes,
    sizes,
    densities,
    warmupIterations,
    measuredIterations,
    casesPerConfiguration,
    targetSampleMilliseconds,
    selectedBackends,
  }), [operation, formats, scalarTypes, sizes, densities, warmupIterations, measuredIterations, casesPerConfiguration, targetSampleMilliseconds, selectedBackends]);

  const availableBackends = useMemo(() => {
    if (executionWorld === "server") {
      return NATIVE_BENCHMARK_BACKENDS.map((backend) => ({
        ...backend,
        available: serverBackendAvailability[backend.id] ?? backend.available ?? false,
      }));
    }
    return [...BENCHMARK_BACKENDS.map((backend) => backend.id === "trace-rust-wasm" ? { ...backend, available: wasmAvailable } : backend), ...uploadedBackends];
  }, [executionWorld, uploadedBackends, serverBackendAvailability, wasmAvailable]);
  const worldRuns = useMemo(() => runs.filter((run) => (run.environment.world ?? "browser") === executionWorld), [runs, executionWorld]);
  const activeRun = worldRuns.find((run) => run.id === activeRunId) ?? worldRuns[0] ?? null;
  const ownBackendId = activeRun?.selectedBackends.find((backendId) => ownBackendIds.includes(backendId)) ?? ownBackendIds[0];
  const activeRunResults = useMemo(() => activeRun ? results.filter((result) => result.runId === activeRun.id) : [], [activeRun, results]);
  const ownResults = useMemo(() => activeRunResults.filter((result) => result.backendId === ownBackendId && result.status === "completed"), [activeRunResults, ownBackendId]);
  const comparisonBackendIds = activeRun ? (comparisonSelections[activeRun.id] ?? activeRun.selectedBackends.filter((backendId) => !ownBackendIds.includes(backendId))) : [];
  const generatedCases = useMemo(() => generateBenchmarkCases(configuration), [configuration]);

  const summary = useMemo(() => ({
    runs: runs.length,
    completedResults: results.filter((result) => result.status === "completed").length,
    median: activeRunResults.length ? average(activeRunResults, (result) => result.statistics.p50) : 0,
    bestGflops: activeRunResults.length ? Math.max(...activeRunResults.map((result) => result.estimatedGflops)) : 0,
  }), [runs.length, results, activeRunResults]);

  const filteredHistory = useMemo(() => results.filter((result) =>
    (historyBackendFilter === "all" || result.backendId === historyBackendFilter)
    && (historyRunFilter === "all" || result.runId === historyRunFilter)
    && (historyOperationFilter === "all" || result.operation === historyOperationFilter),
  ), [results, historyBackendFilter, historyRunFilter, historyOperationFilter]);

  const visibleHistory = filteredHistory.slice((historyPage - 1) * HISTORY_PAGE_SIZE, historyPage * HISTORY_PAGE_SIZE);
  const previewCases = generatedCases.slice((previewPage - 1) * TABLE_PAGE_SIZE, previewPage * TABLE_PAGE_SIZE);
  const visibleOwnResults = ownResults.slice((ownPage - 1) * TABLE_PAGE_SIZE, ownPage * TABLE_PAGE_SIZE);

  const chartSliceOptions = useMemo<ChartSliceOption[]>(() => {
    const options = new Map<string, ChartSliceOption>();
    const traceResults = activeRunResults.filter((result) => result.backendId === ownBackendId && result.status === "completed");

    for (const result of traceResults) {
      const hasCompetitor = comparisonBackendIds.some((backendId) => activeRunResults.some((candidate) =>
        candidate.backendId === backendId
        && candidate.status === "completed"
        && candidate.operation === result.operation
        && candidate.scalarType === result.scalarType
        && candidate.matrixFormat === result.matrixFormat
        && Math.abs(candidate.densityRequested - result.densityRequested) < 1e-12,
      ));
      if (!hasCompetitor) continue;

      const key = `${result.operation}|${result.matrixFormat}|${result.scalarType}|${result.densityRequested}`;
      options.set(key, {
        key,
        operation: result.operation,
        matrixFormat: result.matrixFormat,
        scalarType: result.scalarType,
        densityRequested: result.densityRequested,
      });
    }

    return [...options.values()];
  }, [activeRunResults, comparisonBackendIds, ownBackendId]);

  useEffect(() => {
    localStorage.setItem("trace-benchmark-theme", theme);
  }, [theme]);

  useEffect(() => {
    localStorage.setItem("trace-benchmark-world", executionWorld);
    setActiveRunId(null);
    setSelectedBackends(executionWorld === "server" ? ["trace-rust-native-cpu"] : ["trace-rust-wasm", "trace-core", "javascript-baseline", "ml-matrix", "ndarray-blas-gemv"]);
    if (executionWorld === "server") setScalarTypes(["float64"]);
  }, [executionWorld]);

  useEffect(() => {
    if (executionWorld === "browser" && !wasmAvailable) {
      setSelectedBackends((current) => current.filter((backendId) => backendId !== "trace-rust-wasm"));
    }
    if (executionWorld === "browser" && wasmAvailable) {
      setSelectedBackends((current) => current.includes("trace-rust-wasm") ? current : ["trace-rust-wasm", ...current]);
    }
  }, [executionWorld, wasmAvailable]);

  useEffect(() => {
    localStorage.setItem("trace-benchmark-server-url", serverUrl);
  }, [serverUrl]);

  useEffect(() => {
    localStorage.setItem("trace-benchmark-server-policy", serverExecutionPolicy);
    if (executionWorld === "server") {
      setSelectedBackends(serverExecutionPolicy === "gpu" ? ["trace-rust-native-gpu"] : ["trace-rust-native-cpu"]);
    }
  }, [serverExecutionPolicy, executionWorld]);

  useEffect(() => {
    if (executionWorld !== "browser") {
      setWasmAvailable(false);
      return;
    }
    const controller = new AbortController();
    fetch("/wasm/trace_numerical_wasm.js", { method: "HEAD", signal: controller.signal })
      .then((response) => setWasmAvailable(response.ok))
      .catch(() => setWasmAvailable(false));
    return () => controller.abort();
  }, [executionWorld]);

  useEffect(() => {
    if (executionWorld !== "server") {
      setServerConnected(false);
      setServerBackendAvailability({});
      return;
    }
    const controller = new AbortController();
    fetch(`${serverUrl.replace(/\/$/, "")}/v1/capabilities`, { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json() as Promise<{ nativeBackends?: Array<{ id: string; available: boolean }> }>;
      })
      .then((payload) => {
        const availability: Record<string, boolean> = {};
        for (const backend of payload.nativeBackends ?? []) availability[backend.id] = backend.available;
        setServerBackendAvailability(availability);
        setServerConnected(true);
      })
      .catch(() => {
        if (!controller.signal.aborted) {
          setServerConnected(false);
          setServerBackendAvailability({});
        }
      });
    return () => controller.abort();
  }, [executionWorld, serverUrl]);

  useEffect(() => {
    localStorage.setItem(UPLOADED_ADAPTER_STORAGE_KEY, JSON.stringify(uploadedBackends));
  }, [uploadedBackends]);

  useEffect(() => {
    void loadBenchmarkHistory().then((history) => {
      const orderedRuns = [...history.runs].sort((left, right) => right.startedAt.localeCompare(left.startedAt));
      setRuns(orderedRuns);
      setResults([...history.results].sort((left, right) => right.timestamp.localeCompare(left.timestamp)));
      if (orderedRuns[0]) {
        setActiveRunId(orderedRuns[0].id);
        setComparisonSelections({ [orderedRuns[0].id]: orderedRuns[0].selectedBackends.filter((backendId) => !ownBackendIds.includes(backendId)) });
      }
    });
  }, []);

  useEffect(() => {
    setHistoryPage(1);
  }, [historyBackendFilter, historyRunFilter, historyOperationFilter]);

  useEffect(() => {
    setPreviewPage(1);
    setOwnPage(1);
    setComparisonPage(1);
  }, [operation, formats, scalarTypes, sizes, densities, activeRunId, comparisonBackendIds.join(",")]);

  useEffect(() => {
    setSelectedBackends((current) => current.filter((backendId) => {
      if (backendId === "trace-core") return true;
      const backend = availableBackends.find((candidate) => candidate.id === backendId);
      if (!backend?.operations.includes(operation)) return false;
      if (!matrixOperation) return true;
      const activeFormats: MatrixFormat[] = operation === "matrixVectorMultiply" ? formats : ["dense"];
      return activeFormats.some((format) => backend.matrixFormats.includes(format));
    }));
  }, [operation, formats, availableBackends, matrixOperation]);

  useEffect(() => {
    if (!chartSliceOptions.length) {
      setChartSliceKey("");
      return;
    }
    if (!chartSliceOptions.some((option) => option.key === chartSliceKey)) {
      setChartSliceKey(chartSliceOptions[0].key);
    }
  }, [chartSliceOptions, chartSliceKey]);

  /**
   * Cambia de campaña histórica y restringe los comparadores al contenido real
   * de esa campaña.
   */
  function handleRunChange(runId: string): void {
    setActiveRunId(runId);
    const run = runs.find((candidate) => candidate.id === runId);
    if (run) {
      setComparisonSelections((current) => current[run.id]
        ? current
        : { ...current, [run.id]: run.selectedBackends.filter((backendId) => !ownBackendIds.includes(backendId)) });
    }
  }

  useEffect(() => {
    return () => {
      benchmarkWorkerRef.current?.terminate();
      benchmarkWorkerRef.current = null;
    };
  }, []);

  /**
   * Ejecuta un banco y registra una campaña inmutable.
   */
  async function handleRun(): Promise<void> {
    if (executionWorld === "server") {
      if (!serverConnected) {
        window.alert(t("Connect to the native benchmark server before starting a server campaign."));
        return;
      }
      setRunning(true);
      const serverCases = generateBenchmarkCases(configuration);
      setProgress({ completed: 0, total: serverCases.length * selectedBackends.length });
      try {
        const snapshot = await runServerBenchmarkBank(configuration, { serverUrl, execution: serverExecutionPolicy }, (completed, total) => setProgress({ completed, total }));
        await saveBenchmarkSnapshot(snapshot.run, snapshot.results);
        setRuns((current) => [snapshot.run, ...current.filter((run) => run.id !== snapshot.run.id)]);
        setResults((current) => [...snapshot.results, ...current.filter((result) => result.runId !== snapshot.run.id)]);
        setActiveRunId(snapshot.run.id);
        setComparisonSelections((current) => ({ ...current, [snapshot.run.id]: snapshot.run.selectedBackends.filter((backendId) => backendId !== "trace-rust-native-cpu" && backendId !== "trace-rust-native-gpu") }));
      } catch (error) {
        window.alert(error instanceof Error ? error.message : String(error));
      } finally {
        setRunning(false);
      }
      return;
    }
    if (!selectedBackends.some((backendId) => ownBackendIds.includes(backendId))) {
      const ownBackend = wasmAvailable ? "trace-rust-wasm" : "trace-core";
      setSelectedBackends([ownBackend, ...selectedBackends]);
      return;
    }

    setRunning(true);
    setProgress({ completed: 0, total: generatedCases.length * selectedBackends.length });

    const worker = new Worker(new URL("./benchmark.worker.ts", import.meta.url), { type: "module" });
    benchmarkWorkerRef.current = worker;

    try {
      const snapshot = await new Promise<BenchmarkSnapshot>((resolve, reject) => {
        benchmarkCancelRef.current = () => reject(new Error("Benchmark run cancelled by the user."));

        worker.onmessage = (event: MessageEvent<{ type: string; completed?: number; total?: number; snapshot?: BenchmarkSnapshot; errorMessage?: string }>) => {
          if (event.data.type === "progress") {
            setProgress({ completed: event.data.completed ?? 0, total: event.data.total ?? 0 });
            return;
          }

          if (event.data.type === "completed" && event.data.snapshot) {
            resolve(event.data.snapshot);
            return;
          }

          if (event.data.type === "failed") {
            reject(new Error(event.data.errorMessage ?? "Benchmark worker failed."));
          }
        };

        worker.onerror = (event) => {
          reject(new Error(event.message || "Benchmark worker failed."));
        };

        worker.postMessage({ type: "run", configuration, uploadedBackends });
      });

      await saveBenchmarkSnapshot(snapshot.run, snapshot.results);
      setRuns((current) => [snapshot.run, ...current]);
      setResults((current) => [...snapshot.results, ...current].sort((left, right) => right.timestamp.localeCompare(left.timestamp)));
      setActiveRunId(snapshot.run.id);
      setComparisonSelections((current) => ({ ...current, [snapshot.run.id]: snapshot.run.selectedBackends.filter((backendId) => !ownBackendIds.includes(backendId)) }));
      setActiveTab("benchmark");
    } catch (error) {
      window.alert(`Benchmark failed: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      benchmarkCancelRef.current = null;
      worker.terminate();
      benchmarkWorkerRef.current = null;
      setRunning(false);
    }
  }

  /**
   * Cancela el worker sin bloquear la interfaz ni persistir una campaña parcial.
   */
  function handleCancelRun(): void {
    benchmarkCancelRef.current?.();
    benchmarkWorkerRef.current?.terminate();
    benchmarkWorkerRef.current = null;
    benchmarkCancelRef.current = null;
    setRunning(false);
    setProgress((current) => ({ ...current, completed: 0 }));
  }

  /**
   * Registra un adapter y lo incorpora inmediatamente a la próxima batería.
   */
  function handleAddUploadedAdapter(adapter: UploadedBenchmarkAdapterDefinition): void {
    setUploadedBackends((current) => [...current, adapter]);
    setSelectedBackends((current) => [...current, adapter.id]);
  }

  /**
   * Elimina un adapter local y evita que quede seleccionado en futuros bancos.
   */
  function handleRemoveUploadedAdapter(adapterId: string): void {
    setUploadedBackends((current) => current.filter((adapter) => adapter.id !== adapterId));
    setSelectedBackends((current) => current.filter((backendId) => backendId !== adapterId));
  }

  /**
   * Elimina resultados seleccionados y actualiza runs que hayan perdido todos
   * sus resultados.
   */
  async function handleDeleteSelected(): Promise<void> {
    if (!selectedResultIds.size) return;
    const ids = [...selectedResultIds];
    await deleteBenchmarkResults(ids);
    setResults((current) => current.filter((result) => !selectedResultIds.has(result.id)));
    setSelectedResultIds(new Set());
  }

  /**
   * Limpia todo el histórico persistido.
   */
  async function handleClearHistory(): Promise<void> {
    if (!results.length || !window.confirm(`Delete all ${runs.length} benchmark campaigns?`)) return;
    await clearBenchmarkHistory();
    setRuns([]);
    setResults([]);
    setActiveRunId(null);
    setComparisonSelections({});
    setSelectedResultIds(new Set());
  }

  const historyColumns: TableColumn<BenchmarkResult>[] = [
    { key: "timestamp", header: t("Date"), sortable: true, getSortValue: (row) => row.timestamp, render: (row) => new Date(row.timestamp).toLocaleTimeString() },
    { key: "backend", header: t("Library"), sortable: true, getSortValue: (row) => row.backendName, render: (row) => <span className={ownBackendIds.includes(row.backendId) ? "library-badge own" : "library-badge"} style={{ "--badge-color": getBackendColor(row.backendId) } as CSSProperties}>{row.backendName}</span> },
    { key: "operation", header: t("Operation"), sortable: true, getSortValue: (row) => row.operation, render: (row) => operationLabel(row.operation) },
    { key: "format", header: t("Format"), sortable: true, getSortValue: (row) => row.matrixFormat, render: (row) => row.matrixFormat },
    { key: "size", header: t("Size"), sortable: true, getSortValue: (row) => row.rows * row.columns, render: (row) => <NumericValue value={`${row.rows}×${row.columns}`} explanation={t("Problem size explanation")} /> },
    { key: "density", header: t("Density"), sortable: true, getSortValue: (row) => row.densityRequested, render: (row) => <NumericValue value={`${(row.densityRequested * 100).toFixed(1)}%`} explanation={t("Density explanation")} /> },
    { key: "p50", header: <>{t("P50")} <InfoTooltip text={t("P50 explanation")} /></>, sortable: true, getSortValue: (row) => row.statistics.p50, render: (row) => <NumericValue className="mono" value={formatMilliseconds(row.statistics.p50)} explanation={t("P50 explanation")} /> },
    { key: "gflops", header: <>{t("GFLOP/s")} <InfoTooltip text={t("GFLOP/s explanation")} /></>, sortable: true, getSortValue: (row) => row.estimatedGflops, render: (row) => <NumericValue className="mono" value={row.estimatedGflops.toFixed(3)} explanation={t("GFLOP/s explanation")} /> },
  ];

  const comparisonRows = useMemo<ComparisonRow[]>(() => {
    const byCase = new Map<string, Map<BenchmarkBackendId, BenchmarkResult>>();
    for (const result of activeRunResults) {
      if (result.status !== "completed") continue;
      if (!byCase.has(result.caseId)) byCase.set(result.caseId, new Map());
      byCase.get(result.caseId)!.set(result.backendId, result);
    }

    const rows: ComparisonRow[] = [];
    for (const [caseId, byBackend] of byCase.entries()) {
      const trace = byBackend.get(ownBackendId);
      if (!trace) continue;
      for (const backendId of comparisonBackendIds) {
        const competitor = byBackend.get(backendId);
        if (!competitor) continue;
        rows.push({
          id: `${caseId}:${backendId}`,
          caseId,
          operation: operationLabel(trace.operation),
          size: `${trace.rows}×${trace.columns}`,
          format: trace.matrixFormat,
          density: `${(trace.densityRequested * 100).toFixed(1)}%`,
          traceP50: trace.statistics.p50,
          backendId,
          backendName: competitor.backendName,
          competitorP50: competitor.statistics.p50,
          relative: competitor.statistics.p50 / trace.statistics.p50,
          traceP50Explanation: t("P50 explanation"),
          competitorP50Explanation: t("P50 explanation"),
          relativeExplanation: t("Relative time explanation"),
        });
      }
    }
    return rows;
  }, [activeRunResults, comparisonBackendIds]);

  const visibleComparisonRows = comparisonRows.slice((comparisonPage - 1) * TABLE_PAGE_SIZE, comparisonPage * TABLE_PAGE_SIZE);

  const comparisonColumns: TableColumn<ComparisonRow>[] = [
    { key: "caseId", header: t("Case"), sortable: true, getSortValue: (row) => row.caseId, render: (row) => <span className="mono">{row.caseId}</span> },
    { key: "operation", header: t("Operation"), sortable: true, getSortValue: (row) => row.operation, render: (row) => row.operation },
    { key: "size", header: t("Size"), sortable: true, getSortValue: (row) => row.size, render: (row) => <NumericValue value={row.size} explanation={t("Problem size explanation")} /> },
    { key: "format", header: t("Format"), render: (row) => row.format },
    { key: "density", header: t("Density"), sortable: true, getSortValue: (row) => row.density, render: (row) => <NumericValue value={row.density} explanation={t("Density explanation")} /> },
    { key: "traceP50", header: <>{t("TRACE median time")} <InfoTooltip text={t("P50 explanation")} /></>, sortable: true, getSortValue: (row) => row.traceP50, render: (row) => <NumericValue className="metric-own mono" value={formatMilliseconds(row.traceP50)} explanation={row.traceP50Explanation} /> },
    { key: "backend", header: t("Compared library"), sortable: true, getSortValue: (row) => row.backendName, render: (row) => <span className="library-badge competitor" style={{ "--badge-color": getBackendColor(row.backendId) } as CSSProperties}>{row.backendName}</span> },
    { key: "competitorP50", header: <>{t("Compared median time")} <InfoTooltip text={t("P50 explanation")} /></>, sortable: true, getSortValue: (row) => row.competitorP50, render: (row) => <NumericValue className="mono" value={formatMilliseconds(row.competitorP50)} explanation={row.competitorP50Explanation} /> },
    { key: "relative", header: <>{t("Relative time (Compared / TRACE)")} <InfoTooltip text={t("Relative time explanation")} /></>, sortable: true, getSortValue: (row) => row.relative, render: (row) => <NumericValue className={row.relative < 1 ? "ratio faster" : "ratio slower"} value={formatRatio(row.relative)} explanation={row.relativeExplanation} /> },
  ];

  const selectedChartSlice = chartSliceOptions.find((option) => option.key === chartSliceKey) ?? chartSliceOptions[0] ?? null;
  const chartResults = selectedChartSlice
    ? activeRunResults.filter((result) =>
      result.status === "completed"
      && result.operation === selectedChartSlice.operation
      && result.scalarType === selectedChartSlice.scalarType
      && result.matrixFormat === selectedChartSlice.matrixFormat
      && Math.abs(result.densityRequested - selectedChartSlice.densityRequested) < 1e-12,
    )
    : [];
  const runtimeSeries = buildSeries(chartResults, "runtime", (backendId) => getBackendColor(backendId));
  const gflopsSeries = buildSeries(chartResults, "gflops", (backendId) => getBackendColor(backendId));
  const memorySeries = buildSeries(chartResults, "memory", (backendId) => getBackendColor(backendId));
  const cvSeries = buildSeries(chartResults, "cv", (backendId) => getBackendColor(backendId));
  const tailRatioSeries = buildSeries(chartResults, "tailRatio", (backendId) => getBackendColor(backendId));
  const confidenceIntervals = new Map<string, { low: number; high: number }>();
  for (const result of chartResults) {
    confidenceIntervals.set(`${result.backendName}|${result.rows}`, {
      low: Math.max(0, result.statistics.p50 - result.statistics.confidenceInterval95),
      high: result.statistics.p50 + result.statistics.confidenceInterval95,
    });
  }

  const speedupValues = comparisonBackendIds.map((backendId) => {
    const traceResults = chartResults.filter((result) => result.backendId === ownBackendId);
    const competitor = chartResults.filter((result) => result.backendId === backendId);
    const traceP50 = average(traceResults, (result) => result.statistics.p50);
    const competitorP50 = average(competitor, (result) => result.statistics.p50);
    return {
      label: availableBackends.find((backend) => backend.id === backendId)?.name ?? backendId,
      value: traceP50 > 0 ? competitorP50 / traceP50 : 0,
      color: getBackendColor(backendId),
    };
  });

  const tabItems = [
    { id: "benchmark", label: t("Benchmark"), badge: activeRun ? `${ownResults.length}` : "0" },
    { id: "comparison", label: t("Compare"), badge: activeRun ? `${comparisonBackendIds.length}` : "0" },
    { id: "charts", label: t("Charts"), badge: chartSliceOptions.length ? `${chartSliceOptions.length}` : "0" },
    { id: "history", label: t("History"), badge: `${runs.length}` },
    { id: "methodology", label: t("Methodology") },
    { id: "environment", label: t("Environment") },
  ];

  return (
    <main className={`app-shell theme-${theme}`}>
      <header className="topbar">
        <div className="brand-block">
          <div className="eyebrow">TRACE HPC BENCHMARK LAB · v0.5.0</div>
          <h1>{t("Vector & Matrix Benchmark Lab")}</h1>
          <p>{t("Reproducible numerical performance laboratory")} · <strong>{activeRun ? runLabel(activeRun) : t("No campaign selected")}</strong></p>
        </div>
        <div className="topbar-actions">
          {activeRun ? <span className="campaign-badge"><span className="campaign-dot" />{t("Campaign")} {worldRuns.findIndex((run) => run.id === activeRun.id) + 1} · {activeRun.selectedBackends.length} {t("backends")}</span> : null}
          <LanguageToggle />
          <ThemeToggle theme={theme} onChange={setTheme} />
          <div className="status-badge"><span className={running ? "status-dot running" : "status-dot"} />{running ? t("Running") : executionWorld === "server" ? (serverConnected ? t("Server connected") : t("Server offline")) : t("Ready")}</div>
        </div>
      </header>

      <div className="metrics-grid">
        <MetricCard label={t("Benchmark campaigns")} value={String(worldRuns.length)} detail={t("independent historical runs")} />
        <MetricCard label={t("Completed results")} value={String(summary.completedResults)} detail={t("persisted measurements")} />
        <MetricCard label={t("Active median")} value={summary.median ? formatMilliseconds(summary.median) : "—"} detail={t("P50 across active campaign")} />
        <MetricCard label={t("Peak GFLOP/s")} value={summary.bestGflops ? summary.bestGflops.toFixed(3) : "—"} detail={t("active campaign")} />
      </div>

      <div className="lab-shell">
        <aside className="sidebar">
          <div className="sidebar-heading"><span className="section-kicker">01 / TEST BANK</span><span className="pill">{t("always visible")}</span></div>
          <ExecutionWorldSelector world={executionWorld} serverUrl={serverUrl} serverExecutionPolicy={serverExecutionPolicy} onWorldChange={setExecutionWorld} onServerUrlChange={setServerUrl} onServerExecutionPolicyChange={setServerExecutionPolicy} />
          <div className="operation-browser">
            <div className="operation-category-tabs">{Object.keys(operationCategories).map((category) => {
              const typedCategory = category as OperationCategory;
              return <button key={typedCategory} type="button" className={operationCategory === typedCategory ? "operation-category-tab active" : "operation-category-tab"} onClick={() => { setOperationCategory(typedCategory); const first = operationCategories[typedCategory]?.[0]; if (first) setOperation(first.id); }}>{t(OPERATION_CATEGORY_LABELS[typedCategory])}</button>;
            })}</div>
            <label className="operation-field">{t("Operation")}<select value={operation} onChange={(event) => setOperation(event.target.value as OperationName)}>{(operationCategories[operationCategory] ?? []).map((spec) => <option key={spec.id} value={spec.id} disabled={spec.status === "planned"}>{operationLabel(spec.id)} · {spec.complexity}{spec.status === "planned" ? ` · ${t("planned")}` : ""}</option>)}</select></label>
            {selectedOperationSpec ? <div className="operation-description"><strong>{selectedOperationSpec.formula}</strong><span>{selectedOperationSpec.complexity}</span>{selectedOperationSpec.status === "planned" ? <em>{t("Planned operation")}</em> : null}</div> : null}
          </div>

          <div className="sidebar-grid">
            {matrixOperation ? (
              <fieldset><legend>{t("Format")}</legend><div className="check-grid compact-grid">{(["dense", "csr", "csc", "ellpack"] as MatrixFormat[]).map((format) => <label className="check-row" key={format}><input type="checkbox" checked={formats.includes(format)} disabled={operation !== "matrixVectorMultiply" && format !== "dense"} onChange={() => toggleValue(formats, format, setFormats)} />{format === "dense" ? t("Dense") : format.toUpperCase()}</label>)}</div></fieldset>
            ) : null}
            <fieldset><legend>{t("Precision")}</legend><div className="check-grid compact-grid">{(["float32", "float64"] as ScalarType[]).map((precision) => <label className="check-row" key={precision}><input type="checkbox" checked={scalarTypes.includes(precision)} disabled={executionWorld === "server" && precision !== "float64"} onChange={() => toggleValue(scalarTypes, precision, setScalarTypes)} />{precision === "float64" ? "Float64" : "Float32"}</label>)}</div></fieldset>
            <fieldset className="span-two"><legend>{t("Problem sizes")}</legend><div className="size-grid">{SIZE_OPTIONS.map((size) => <label className="check-row" key={size}><input type="checkbox" checked={sizes.includes(size)} onChange={() => toggleValue(sizes, size, setSizes)} />{size}</label>)}</div></fieldset>
            {matrixOperation ? <fieldset className="span-two"><legend>{t("Requested density")}</legend><div className="density-grid">{[1, 0.1, 0.01].map((density) => <label className="check-row" key={density}><input type="checkbox" checked={densities.includes(density)} onChange={() => toggleValue(densities, density, setDensities)} />{density * 100}%</label>)}</div></fieldset> : null}
          </div>

          <BackendSelector
            backends={availableBackends}
            selected={selectedBackends}
            operation={operation}
            matrixFormats={operation === "matrixVectorMultiply" ? formats : matrixOperation ? ["dense"] : []}
            onChange={(next) => setSelectedBackends(next.some((backendId) => ownBackendIds.includes(backendId)) ? next : [ownBackendId, ...next])}
          />

          {executionWorld === "browser" ? <AdapterUpload adapters={uploadedBackends} onAdd={handleAddUploadedAdapter} onRemove={handleRemoveUploadedAdapter} /> : null}

          <div className="compact-parameters">
            <label>{t("Warmup")}<input type="number" min="0" max="50" value={warmupIterations} onChange={(event) => setWarmupIterations(Number(event.target.value))} /></label>
            <label>{t("Samples")}<input type="number" min="5" max="100" value={measuredIterations} onChange={(event) => setMeasuredIterations(Number(event.target.value))} /></label>
            <label>{t("Cases / config")}<input type="number" min="1" max="10" value={casesPerConfiguration} onChange={(event) => setCasesPerConfiguration(Number(event.target.value))} /></label>
            <label>{t("Target / sample")}<input type="number" min="1" max="100" value={targetSampleMilliseconds} onChange={(event) => setTargetSampleMilliseconds(Number(event.target.value))} /></label>
          </div>

          {running ? (
            <div className="run-actions">
              <button className="primary-button" type="button" disabled>{t("Running test bank…")}</button>
              <button className="secondary-button cancel-button" type="button" onClick={handleCancelRun}>{t("Cancel run")}</button>
            </div>
          ) : (
            <button className="primary-button" type="button" disabled={selectedOperationSpec?.status === "planned" || sizes.length === 0 || scalarTypes.length === 0 || selectedBackends.length === 0 || (operation === "matrixVectorMultiply" && formats.length === 0)} onClick={handleRun}>
              {executionWorld === "server" ? t("Server mode: run from native server") : t("Generate & run test bank")}
            </button>
          )}
          <div className="progress"><div className="progress-label"><span>{progress.completed} / {progress.total || generatedCases.length * selectedBackends.length}</span><span>{progress.total ? `${Math.round((progress.completed / progress.total) * 100)}%` : t("Ready")}</span></div><div className="progress-track"><div className="progress-value" style={{ width: `${progress.total ? (progress.completed / progress.total) * 100 : 0}%` }} /></div></div>
          <p className="sidebar-note">{t("Each test bank keeps its own library selection. A campaign can never display a library that it did not execute.")}</p>
        </aside>

        <section className="workspace-panel">
          <Tabs items={tabItems} activeId={activeTab} onChange={(id) => setActiveTab(id as MainTab)} />

          <div className="workspace-content">
            {!activeRun && activeTab !== "methodology" && activeTab !== "environment" ? (
              <section className="empty-workspace"><span className="section-kicker">{t("NO CAMPAIGN")}</span><h2>{t("Run the first benchmark bank")}</h2><p>{t("Select the test battery in the left panel and run the bank. Charts and comparisons use real measurements only.")}</p></section>
            ) : null}

            {activeTab === "benchmark" && activeRun ? (
              <div className="tab-content">
                <section className="panel campaign-selector-panel">
                  <div className="campaign-selector-heading">
                    <div><span className="section-kicker">02 / BENCHMARK CAMPAIGN</span><h2>{t("Historical test battery")}</h2><p>{t("Select any previously executed test battery. The latest campaign is shown by default.")}</p></div>
                    <span className="library-badge own large" style={{ "--badge-color": getBackendColor(ownBackendId) } as CSSProperties}>{t("TRACE · OUR LIBRARY")}</span>
                  </div>
                  <label>{t("Test battery / campaign")}<select value={activeRun.id} onChange={(event) => handleRunChange(event.target.value)}>{worldRuns.map((run) => <option key={run.id} value={run.id}>{new Date(run.startedAt).toLocaleString()} · {operationLabel(run.configuration.operation)} · {run.configuration.sizes.join("/")}</option>)}</select></label>
                  <div className="campaign-facts"><span>{activeRun.caseCount} {t("cases")}</span><span>{activeRun.resultCount} {t("measurements")}</span><span>{activeRun.selectedBackends.length} {t("executed libraries")}</span><span>{activeRun.configuration.scalarTypes.join(", ")}</span></div>
                </section>
                <div className="compact-metrics"><MetricCard label={t("P50")} value={ownResults.length ? formatMilliseconds(average(ownResults, (result) => result.statistics.p50)) : "—"} explanation={t("P50 explanation")} /><MetricCard label={t("P95")} value={ownResults.length ? formatMilliseconds(average(ownResults, (result) => result.statistics.p95)) : "—"} explanation={t("P95 explanation")} /><MetricCard label={t("Average CV")} value={ownResults.length ? `${(average(ownResults, (result) => result.statistics.coefficientOfVariation) * 100).toFixed(1)}%` : "—"} explanation={t("CV explanation")} /><MetricCard label={t("Working set")} value={ownResults.length ? formatBytes(Math.max(...ownResults.map((result) => result.estimatedMemoryBytes))) : "—"} explanation={t("Working set explanation")} /></div>
                <section className="panel compact-panel"><div className="panel-heading"><div><span className="section-kicker">{t("ACTIVE CAMPAIGN")}</span><h2>{activeRun.label}</h2></div><span className="pill">{activeRun.resultCount} results · {activeRun.caseCount} {t("cases")}</span></div><div className="configuration-summary"><div><span>{t("Operation")}</span><strong>{operationLabel(activeRun.configuration.operation)}</strong></div><div><span>{t("Sizes")}</span><strong>{activeRun.configuration.sizes.join(", ")}</strong></div><div><span>{t("Precision")}</span><strong>{activeRun.configuration.scalarTypes.join(", ")}</strong></div><div><span>{t("Formats")}</span><strong>{activeRun.configuration.matrixFormats.join(", ")}</strong></div><div><span>{t("Density")}</span><strong>{activeRun.configuration.densities.map((density) => `${density * 100}%`).join(", ")}</strong></div><div><span>{t("Libraries")}</span><strong>{activeRun.selectedBackends.length}</strong></div></div></section>
                <section className="panel table-panel"><div className="panel-heading"><div><span className="section-kicker">{t("MEASUREMENTS")}</span><h2>{t("TRACE results")} · {ownResults.length} {t("measurements")}</h2></div><button className="secondary-button" type="button" onClick={() => downloadText(`trace-${activeRun.id}.json`, JSON.stringify({ run: activeRun, results: ownResults }, null, 2), "application/json")}>{t("Export")}</button></div><DataTable rows={visibleOwnResults.map((result) => ({ ...result, id: result.id }))} columns={historyColumns.filter((column) => ["backend", "operation", "format", "size", "density", "p50", "gflops"].includes(column.key))} emptyMessage="No TRACE results in this campaign." /><Pagination currentPage={ownPage} totalItems={ownResults.length} pageSize={TABLE_PAGE_SIZE} onPageChange={setOwnPage} /></section>
              </div>
            ) : null}

            {activeTab === "comparison" && activeRun ? (
              <div className="tab-content">
                <section className="panel hero-panel"><div><span className="section-kicker">03 / COMPARISON</span><h2>{t("Same problem · same campaign")}</h2><p>{t("TRACE is always the baseline. You can only select competitors that were actually executed in the active campaign.")}</p></div><span className="pill">{comparisonRows.length} {t("comparable pairs")}</span></section>
                <div className="comparison-campaign-select"><label>{t("Test battery / campaign")}<select value={activeRun.id} onChange={(event) => handleRunChange(event.target.value)}>{worldRuns.map((run) => <option key={run.id} value={run.id}>{new Date(run.startedAt).toLocaleString()} · {operationLabel(run.configuration.operation)} · {run.caseCount} {t("cases")}</option>)}</select></label></div>
                <section className="panel compact-panel"><div className="panel-heading"><div><span className="section-kicker">{t("EXECUTED LIBRARIES")}</span><h2>{t("What should be visible?")}</h2></div><span className="library-badge own" style={{ "--badge-color": getBackendColor("trace-core") } as CSSProperties}>{t("TRACE · ALWAYS ON")}</span></div><div className="comparison-selector"><span className="backend-chip own"><i style={{ background: getBackendColor(ownBackendId) }} />{availableBackends.find((backend) => backend.id === ownBackendId)?.name ?? "TRACE"}</span>{activeRun.selectedBackends.filter((backendId) => !ownBackendIds.includes(backendId)).map((backendId) => { const backend = availableBackends.find((candidate) => candidate.id === backendId)!; return <label className="backend-chip" key={backendId}><input type="checkbox" checked={comparisonBackendIds.includes(backendId)} onChange={() => setComparisonSelections((current) => ({ ...current, [activeRun.id]: current[activeRun.id]?.includes(backendId) ? current[activeRun.id].filter((id) => id !== backendId) : [...(current[activeRun.id] ?? []), backendId] }))} /><i style={{ background: getBackendColor(backendId) }} /><span>{backend.name}</span></label>; })}{!activeRun.selectedBackends.some((backendId) => !ownBackendIds.includes(backendId)) ? <span className="subtle inline-note">{t("No competitor was executed in this campaign.")}</span> : null}</div></section>
                <section className="panel table-panel"><div className="panel-heading"><div><span className="section-kicker">{t("COMPARISON TABLE")}</span><h2>{t("One row = one comparable library/case")}</h2></div><span className="comparison-key"><b>TRACE</b> {t("own library")} · <b>Compared / TRACE &lt; 1×</b> {t("competitor faster")}</span></div><DataTable rows={visibleComparisonRows} columns={comparisonColumns} emptyMessage={t("Select at least one executed competitor to build the comparison.")} /><Pagination currentPage={comparisonPage} totalItems={comparisonRows.length} pageSize={TABLE_PAGE_SIZE} onPageChange={setComparisonPage} /></section>
              </div>
            ) : null}

            {activeTab === "charts" && activeRun ? (
              <div className="tab-content">
                <section className="panel hero-panel"><div><span className="section-kicker">04 / CHARTS</span><h2>{t("Performance curves")}</h2><p>{t("Now the selected test battery/slice is explicit. Change the slice without running the bank again.")}</p></div><span className="pill">{chartSliceOptions.length} {t("comparable slices")}</span></section>
                <section className="panel chart-controls"><div className="chart-selection-grid"><label>{t("Campaign")}<select value={activeRun.id} onChange={(event) => handleRunChange(event.target.value)}>{worldRuns.map((run) => <option key={run.id} value={run.id}>{new Date(run.startedAt).toLocaleString()} · {operationLabel(run.configuration.operation)}</option>)}</select></label><label>{t("Test slice")}<select value={chartSliceKey} onChange={(event) => setChartSliceKey(event.target.value)}>{chartSliceOptions.map((option) => <option key={option.key} value={option.key}>{operationLabel(option.operation)} · {option.matrixFormat} · {option.scalarType.toUpperCase()} · {option.densityRequested * 100}%</option>)}</select></label><div className="chart-slice-summary"><span>{t("Selected comparison")}</span><strong>{selectedChartSlice ? `${operationLabel(selectedChartSlice.operation)} · ${selectedChartSlice.matrixFormat} · ${selectedChartSlice.scalarType.toUpperCase()} · ${selectedChartSlice.densityRequested * 100}%` : t("No compatible slice")}</strong></div></div><div className="chart-library-legend"><span className="legend-item own"><i className="legend-dot" style={{ background: getBackendColor(ownBackendId) }} />{t("TRACE Matrix Core · OUR LIBRARY")}</span>{comparisonBackendIds.map((backendId) => <span className="legend-item" key={backendId}><i className="legend-dot" style={{ background: getBackendColor(backendId) }} />{availableBackends.find((backend) => backend.id === backendId)?.name}</span>)}</div></section>
                <div className="chart-view-tabs" role="tablist" aria-label="Professional benchmark charts">
                  {([
                    ["overview", t("Overview")],
                    ["stability", t("Stability")],
                    ["memory", t("Memory")],
                    ["statistics", t("Statistics")],
                  ] as const).map(([id, label]) => <button key={id} type="button" className={chartView === id ? "chart-view-tab active" : "chart-view-tab"} onClick={() => setChartView(id)}>{label}</button>)}
                </div>
                <div className="charts-grid">
                  {chartView === "overview" ? <>
                    <LineChart title={t("Runtime vs problem size")} yLabel={t("P50 milliseconds")} series={runtimeSeries} formatY={(value) => `${value.toFixed(2)} ms`} />
                    <LineChart title={t("GFLOP/s vs problem size")} yLabel={t("estimated GFLOP/s")} series={gflopsSeries} formatY={(value) => value.toFixed(2)} />
                    <BarChart title={t("Relative runtime")} values={speedupValues} formatValue={(value) => `${value.toFixed(2)}×`} />
                    <ErrorBarChart title="P50 with confidence interval" series={runtimeSeries} intervals={confidenceIntervals} formatY={(value) => `${value.toFixed(2)} ms`} />
                  </> : null}
                  {chartView === "stability" ? <>
                    <LineChart title={t("Coefficient of variation")} yLabel={t("CV (%)")} series={cvSeries} formatY={(value) => `${value.toFixed(1)}%`} />
                    <LineChart title={t("Tail latency")} yLabel={t("P95 / P50")} series={tailRatioSeries} formatY={(value) => `${value.toFixed(2)}×`} />
                    <ErrorBarChart title={t("P50 + 95% confidence interval")} series={runtimeSeries} intervals={confidenceIntervals} formatY={(value) => `${value.toFixed(2)} ms`} />
                    <BarChart title={t("Relative runtime")} values={speedupValues} formatValue={(value) => `${value.toFixed(2)}×`} />
                  </> : null}
                  {chartView === "memory" ? <>
                    <LineChart title={t("Estimated working set")} yLabel={t("bytes")} series={memorySeries} formatY={(value) => formatBytes(value)} />
                    <LineChart title={t("Runtime vs working set")} yLabel={t("P50 milliseconds")} series={runtimeSeries} formatY={(value) => `${value.toFixed(2)} ms`} />
                    <LineChart title={t("GFLOP/s vs size")} yLabel={t("estimated GFLOP/s")} series={gflopsSeries} formatY={(value) => value.toFixed(2)} />
                    <BarChart title={t("Relative runtime")} values={speedupValues} />
                  </> : null}
                  {chartView === "statistics" ? <>
                    <LineChart title={t("P50 runtime")} yLabel={t("milliseconds")} series={runtimeSeries} formatY={(value) => `${value.toFixed(2)} ms`} />
                    <LineChart title={t("P95 / P50 tail ratio")} yLabel={t("ratio")} series={tailRatioSeries} formatY={(value) => `${value.toFixed(2)}×`} />
                    <LineChart title={t("Coefficient of variation")} yLabel={t("CV (%)")} series={cvSeries} formatY={(value) => `${value.toFixed(1)}%`} />
                    <ErrorBarChart title={t("Confidence interval")} series={runtimeSeries} intervals={confidenceIntervals} formatY={(value) => `${value.toFixed(2)} ms`} />
                  </> : null}
                </div>
              </div>
            ) : null}

            {activeTab === "history" ? (
              <div className="tab-content">
                <section className="panel hero-panel"><div><span className="section-kicker">05 / HISTORY</span><h2>{t("Persistent benchmark campaigns")}</h2><p>{t("The history keeps every campaign and its backends. Navigation uses pages instead of table scrolling.")}</p></div><div className="button-group"><button className="secondary-button" disabled={!results.length} onClick={() => downloadText("trace-benchmark-history.json", JSON.stringify({ runs, results }, null, 2), "application/json")}>{t("Export JSON")}</button><button className="danger-button" disabled={!selectedResultIds.size} onClick={handleDeleteSelected}>Delete ({selectedResultIds.size})</button><button className="danger-button" disabled={!results.length} onClick={handleClearHistory}>{t("Clear all")}</button></div></section>
                <section className="panel table-panel"><div className="filter-bar"><label>{t("Campaign")}<select value={historyRunFilter} onChange={(event) => setHistoryRunFilter(event.target.value)}><option value="all">{t("All campaigns")}</option>{worldRuns.map((run) => <option key={run.id} value={run.id}>{new Date(run.startedAt).toLocaleString()} · {operationLabel(run.configuration.operation)}</option>)}</select></label><label>{t("Library")}<select value={historyBackendFilter} onChange={(event) => setHistoryBackendFilter(event.target.value)}><option value="all">{t("All libraries")}</option>{availableBackends.map((backend) => <option key={backend.id} value={backend.id}>{backend.name}</option>)}</select></label><label>{t("Operation")}<select value={historyOperationFilter} onChange={(event) => setHistoryOperationFilter(event.target.value)}><option value="all">{t("All operations")}</option>{Object.entries(OPERATION_LABELS).map(([value]) => <option key={value} value={value}>{operationLabel(value as OperationName)}</option>)}</select></label><button className="secondary-button" type="button" onClick={() => { setHistoryRunFilter("all"); setHistoryBackendFilter("all"); setHistoryOperationFilter("all"); }}>{t("Reset")}</button></div><div className="history-meta"><span>{filteredHistory.length} matching results</span><span>{worldRuns.length} campaigns</span></div><DataTable rows={visibleHistory.map((result) => ({ ...result, id: result.id }))} columns={historyColumns} selectable selectedIds={selectedResultIds} onSelectionChange={setSelectedResultIds} emptyMessage="No stored results match the current filters." /><Pagination currentPage={historyPage} totalItems={filteredHistory.length} pageSize={HISTORY_PAGE_SIZE} onPageChange={setHistoryPage} /></section>
              </div>
            ) : null}

            {activeTab === "methodology" ? (
              <div className="tab-content"><section className="panel hero-panel"><div><span className="section-kicker">06 / METHODOLOGY</span><h2>{t("Professional benchmark protocol")}</h2><p>{t("The interface separates configuration, measurement, comparison and result presentation.")}</p></div></section><div className="method-grid six"><div><strong>1. Same workload</strong><p>{t("Same dimensions, precision, operation and deterministic data for every backend.")}</p></div><div><strong>2. Preparation excluded</strong><p>{t("Allocation, conversion and adaptation stay outside the timed region.")}</p></div><div><strong>3. Warmup + repetitions</strong><p>{t("Separate warmup, repeated samples and interleaved ordering reduce state bias.")}</p></div><div><strong>4. Stable timing</strong><p>{t("Inner iterations are calibrated to a target duration; every sample stores its metadata.")}</p></div><div><strong>5. Statistical reporting</strong><p>{t("P50/P90/P95/P99, IQR, deviation, CV and CI95 are reported; P95/P50 tail latency is also visualized.")}</p></div><div><strong>6. Auditability</strong><p>{t("Campaign, environment, backend, version, NNZ, density, working set and checksum are recorded.")}</p></div></div></div>
            ) : null}

            {activeTab === "environment" ? (
              <div className="tab-content"><section className="panel hero-panel"><div><span className="section-kicker">07 / ENVIRONMENT</span><h2>{t("Execution environment")}</h2><p>{t("The environment is attached to each campaign so results remain reproducible and interpretable.")}</p></div></section><section className="panel"><div className="configuration-summary environment-grid"><div><span>{t("Browser / UA")}</span><strong>{activeRun?.environment.userAgent ?? readEnvironmentSnapshot().userAgent}</strong></div><div><span>{t("Platform")}</span><strong>{activeRun?.environment.platform ?? readEnvironmentSnapshot().platform}</strong></div><div><span>{t("Logical cores")}</span><strong>{activeRun?.environment.hardwareConcurrency ?? readEnvironmentSnapshot().hardwareConcurrency ?? "—"}</strong></div><div><span>{t("Device memory")}</span><strong>{activeRun?.environment.deviceMemoryGb ? `${activeRun.environment.deviceMemoryGb} GB` : "Not exposed"}</strong></div><div><span>{t("Cross-origin isolated")}</span><strong>{String(activeRun?.environment.crossOriginIsolated ?? window.crossOriginIsolated)}</strong></div><div><span>SharedArrayBuffer</span><strong>{String(activeRun?.environment.sharedArrayBuffer ?? typeof SharedArrayBuffer !== "undefined")}</strong></div></div></section></div>
            ) : null}
          </div>
        </section>
      </div>
    </main>
  );
}
