/**
 * Persistencia versionada del laboratorio.
 *
 * Los runs y sus resultados se almacenan por separado para que una ejecución
 * sea una unidad auditable. Esto evita el problema de mezclar resultados de
 * distintos bancos y permite construir subsolapas de comparación por run.
 */

import type { BenchmarkResult, BenchmarkRun } from "./benchmark";

const DATABASE_NAME = "trace-benchmark-lab";
const DATABASE_VERSION = 2;
const RUN_STORE_NAME = "benchmark-runs";
const RESULT_STORE_NAME = "benchmark-results-v2";
const LEGACY_STORE_NAME = "benchmark-results";

export interface BenchmarkHistory {
  readonly runs: BenchmarkRun[];
  readonly results: BenchmarkResult[];
}

/**
 * Abre IndexedDB y crea los almacenes del modelo de datos versionado.
 */
function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DATABASE_NAME, DATABASE_VERSION);

    request.onupgradeneeded = () => {
      const database = request.result;

      if (!database.objectStoreNames.contains(RUN_STORE_NAME)) {
        const runs = database.createObjectStore(RUN_STORE_NAME, { keyPath: "id" });
        runs.createIndex("startedAt", "startedAt", { unique: false });
      }

      if (!database.objectStoreNames.contains(RESULT_STORE_NAME)) {
        const results = database.createObjectStore(RESULT_STORE_NAME, { keyPath: "id" });
        results.createIndex("runId", "runId", { unique: false });
        results.createIndex("backendId", "backendId", { unique: false });
        results.createIndex("operation", "operation", { unique: false });
        results.createIndex("timestamp", "timestamp", { unique: false });
      }

      // Migración conservadora del histórico v0.2.x: los registros antiguos
      // no tenían runId, por lo que se agrupan en una campaña legacy única.
      if (database.objectStoreNames.contains(LEGACY_STORE_NAME)) {
        const transaction = request.transaction;
        if (!transaction) return;

        const legacyStore = transaction.objectStore(LEGACY_STORE_NAME);
        const legacyRequest = legacyStore.getAll();
        legacyRequest.onsuccess = () => {
          const legacyResults = legacyRequest.result as Array<Record<string, unknown>>;
          if (!legacyResults.length) return;

          const runId = `legacy-${Date.now()}`;
          const first = legacyResults[0];
          const operation = String(first.operation) as BenchmarkRun["configuration"]["operation"];
          const scalarType = String(first.scalarType) as BenchmarkRun["configuration"]["scalarTypes"][number];
          const legacyMatrixFormat = String(first.matrixFormat);
          const matrixFormat = legacyMatrixFormat === "none" ? "dense" : legacyMatrixFormat as BenchmarkRun["configuration"]["matrixFormats"][number];
          const sizes = [...new Set(legacyResults.map((item) => Number(item.rows)).filter(Number.isFinite))];
          const migratedRun: BenchmarkRun = {
            id: runId,
            startedAt: String(first.timestamp ?? new Date().toISOString()),
            completedAt: String(first.timestamp ?? new Date().toISOString()),
            label: "Legacy v0.2.x campaign",
            configuration: {
              operation,
              matrixFormats: [matrixFormat],
              scalarTypes: [scalarType],
              sizes,
              densities: [...new Set(legacyResults.map((item) => Number(item.density)).filter(Number.isFinite))],
              warmupIterations: Number(first.warmupIterations ?? 0),
              measuredIterations: Number(first.measuredIterations ?? 0),
              casesPerConfiguration: 1,
              targetSampleMilliseconds: 1,
              selectedBackends: ["trace-core"],
            },
            selectedBackends: ["trace-core"],
            environment: {
              userAgent: "legacy-v0.2.x",
              platform: "unknown",
              language: "unknown",
              hardwareConcurrency: null,
              deviceMemoryGb: null,
              crossOriginIsolated: false,
              sharedArrayBuffer: false,
              timestamp: String(first.timestamp ?? new Date().toISOString()),
            },
            caseCount: legacyResults.length,
            resultCount: legacyResults.length,
          };

          transaction.objectStore(RUN_STORE_NAME).put(migratedRun);
          const resultStore = transaction.objectStore(RESULT_STORE_NAME);
          for (const legacy of legacyResults) {
            const caseId = String(legacy.id);
            resultStore.put({
              id: `${runId}:trace-core:${caseId}`,
              runId,
              caseId,
              timestamp: String(legacy.timestamp),
              operation: String(legacy.operation),
              matrixFormat: String(legacy.matrixFormat),
              scalarType: String(legacy.scalarType),
              rows: Number(legacy.rows),
              columns: Number(legacy.columns),
              vectorLength: Number(legacy.vectorLength),
              densityRequested: Number(legacy.density),
              densityActual: Number(legacy.density),
              nnz: 0,
              ellpackStorageSlots: 0,
              ellpackPaddingRatio: 0,
              warmupIterations: Number(legacy.warmupIterations ?? 0),
              measuredIterations: Number(legacy.measuredIterations ?? 0),
              innerIterations: 1,
              statistics: {
                min: Number(legacy.minMilliseconds ?? 0),
                p50: Number(legacy.medianMilliseconds ?? 0),
                p90: Number(legacy.medianMilliseconds ?? 0),
                p95: Number(legacy.medianMilliseconds ?? 0),
                p99: Number(legacy.maxMilliseconds ?? 0),
                max: Number(legacy.maxMilliseconds ?? 0),
                mean: Number(legacy.meanMilliseconds ?? 0),
                standardDeviation: Number(legacy.standardDeviationMilliseconds ?? 0),
                coefficientOfVariation: 0,
                iqr: 0,
                confidenceInterval95: 0,
              },
              estimatedFlops: Number(legacy.estimatedFlops ?? 0),
              estimatedGflops: Number(legacy.estimatedGflops ?? 0),
              estimatedMemoryBytes: Number(legacy.estimatedMemoryBytes ?? 0),
              backendId: "trace-core",
              backendName: "TRACE Matrix Core",
              backendVersion: "0.2.x",
              environment: "browser",
              status: "completed",
              checksum: 0,
            });
          }
        };
      }
    };

    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("Unable to open IndexedDB."));
  });
}

/**
 * Lee runs y resultados desde los almacenes de la versión actual.
 */
export async function loadBenchmarkHistory(): Promise<BenchmarkHistory> {
  const database = await openDatabase();

  return new Promise((resolve, reject) => {
    const transaction = database.transaction([RUN_STORE_NAME, RESULT_STORE_NAME], "readonly");
    const runsRequest = transaction.objectStore(RUN_STORE_NAME).getAll();
    const resultsRequest = transaction.objectStore(RESULT_STORE_NAME).getAll();

    transaction.oncomplete = () => {
      database.close();
      resolve({
        runs: runsRequest.result as BenchmarkRun[],
        results: resultsRequest.result as BenchmarkResult[],
      });
    };
    transaction.onerror = () => {
      database.close();
      reject(transaction.error ?? new Error("Unable to load benchmark history."));
    };
  });
}

/**
 * Persiste una ejecución completa en una única transacción.
 */
export async function saveBenchmarkSnapshot(run: BenchmarkRun, results: BenchmarkResult[]): Promise<void> {
  const database = await openDatabase();

  return new Promise((resolve, reject) => {
    const transaction = database.transaction([RUN_STORE_NAME, RESULT_STORE_NAME], "readwrite");
    transaction.objectStore(RUN_STORE_NAME).put(run);

    const resultStore = transaction.objectStore(RESULT_STORE_NAME);
    for (const result of results) resultStore.put(result);

    transaction.oncomplete = () => {
      database.close();
      resolve();
    };
    transaction.onerror = () => {
      database.close();
      reject(transaction.error ?? new Error("Unable to save benchmark snapshot."));
    };
  });
}

/**
 * Elimina runs y todos sus resultados asociados.
 */
export async function deleteBenchmarkRuns(runIds: string[]): Promise<void> {
  if (runIds.length === 0) return;
  const database = await openDatabase();

  return new Promise((resolve, reject) => {
    const transaction = database.transaction([RUN_STORE_NAME, RESULT_STORE_NAME], "readwrite");
    const runStore = transaction.objectStore(RUN_STORE_NAME);
    const resultStore = transaction.objectStore(RESULT_STORE_NAME);
    const runIndex = resultStore.index("runId");

    for (const runId of runIds) {
      runStore.delete(runId);
      const request = runIndex.openCursor(IDBKeyRange.only(runId));
      request.onsuccess = () => {
        const cursor = request.result;
        if (!cursor) return;
        cursor.delete();
        cursor.continue();
      };
    }

    transaction.oncomplete = () => {
      database.close();
      resolve();
    };
    transaction.onerror = () => {
      database.close();
      reject(transaction.error ?? new Error("Unable to delete benchmark runs."));
    };
  });
}

/**
 * Borra toda la colección actual manteniendo la estructura de la base de datos.
 */
export async function clearBenchmarkHistory(): Promise<void> {
  const database = await openDatabase();

  return new Promise((resolve, reject) => {
    const transaction = database.transaction([RUN_STORE_NAME, RESULT_STORE_NAME], "readwrite");
    transaction.objectStore(RUN_STORE_NAME).clear();
    transaction.objectStore(RESULT_STORE_NAME).clear();

    transaction.oncomplete = () => {
      database.close();
      resolve();
    };
    transaction.onerror = () => {
      database.close();
      reject(transaction.error ?? new Error("Unable to clear benchmark history."));
    };
  });
}

/**
 * Elimina resultados concretos y conserva el run si todavía contiene datos.
 */
export async function deleteBenchmarkResults(ids: string[]): Promise<void> {
  if (ids.length === 0) return;
  const database = await openDatabase();

  return new Promise((resolve, reject) => {
    const transaction = database.transaction(RESULT_STORE_NAME, "readwrite");
    const store = transaction.objectStore(RESULT_STORE_NAME);
    for (const id of ids) store.delete(id);

    transaction.oncomplete = () => {
      database.close();
      resolve();
    };
    transaction.onerror = () => {
      database.close();
      reject(transaction.error ?? new Error("Unable to delete benchmark results."));
    };
  });
}
