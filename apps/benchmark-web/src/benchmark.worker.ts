/**
 * Worker de ejecución del banco de benchmarks.
 *
 * El objetivo es sacar completamente los kernels de la UI para que React y el
 * navegador puedan seguir pintando, aceptar interacción y actualizar progreso
 * mientras se ejecutan bancos grandes.
 */

import { runBenchmarkBank, type BenchmarkConfiguration, type BenchmarkSnapshot, type UploadedBenchmarkAdapterDefinition } from "./benchmark";

interface BenchmarkWorkerRequest {
  readonly type: "run";
  readonly configuration: BenchmarkConfiguration;
  readonly uploadedBackends: readonly UploadedBenchmarkAdapterDefinition[];
}

interface BenchmarkWorkerProgress {
  readonly type: "progress";
  readonly completed: number;
  readonly total: number;
}

interface BenchmarkWorkerCompleted {
  readonly type: "completed";
  readonly snapshot: BenchmarkSnapshot;
}

interface BenchmarkWorkerFailed {
  readonly type: "failed";
  readonly errorMessage: string;
}

type BenchmarkWorkerResponse = BenchmarkWorkerProgress | BenchmarkWorkerCompleted | BenchmarkWorkerFailed;

const workerScope = globalThis as typeof globalThis & {
  onmessage: ((event: MessageEvent<BenchmarkWorkerRequest>) => void) | null;
  postMessage: (message: BenchmarkWorkerResponse) => void;
};

/**
 * Ejecuta el banco en segundo plano y publica progreso incremental.
 */
workerScope.onmessage = async (event): Promise<void> => {
  if (event.data.type !== "run") return;

  try {
    const snapshot = await runBenchmarkBank(event.data.configuration, (completed, total) => {
      workerScope.postMessage({ type: "progress", completed, total });
    }, event.data.uploadedBackends);

    workerScope.postMessage({ type: "completed", snapshot });
  } catch (error) {
    workerScope.postMessage({
      type: "failed",
      errorMessage: error instanceof Error ? error.message : String(error),
    });
  }
};
