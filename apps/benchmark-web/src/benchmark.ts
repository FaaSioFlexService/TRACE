/**
 * Motor profesional de generación, ejecución y comparación de benchmarks.
 *
 * Esta capa mantiene separados tres conceptos:
 * 1. caso matemático: el problema que se quiere resolver;
 * 2. backend: la implementación que resuelve el problema;
 * 3. ejecución: una instantánea inmutable que agrupa los resultados de un
 *    banco completo y sus backends seleccionados.
 *
 * Las conversiones y asignaciones se realizan fuera de la región cronometrada.
 * La medición usa varias ejecuciones internas por muestra para evitar que las
 * operaciones pequeñas queden ocultas por la resolución del reloj del browser.
 */

import {
  OPERATION_CATALOG,
  createDenseMatrix,
  createRandomData,
  createVector,
  denseToCsr,
  denseToCsc,
  denseToEllpack,
  matrixMatrixMultiply,
  matrixVectorMultiply,
  vectorAdd,
  vectorNorm2,
  vectorScale,
  dotProduct,
  type DenseMatrix,
  type MatrixFormat,
  type ScalarType,
} from "@trace/vector-matrix-core";
import { Matrix as MlMatrix } from "ml-matrix";
import ndarray from "ndarray";
import ndarrayOps from "ndarray-ops";
import gemv from "ndarray-blas-gemv";

export type OperationName = import("@trace/vector-matrix-core").OperationName;

export type BenchmarkEnvironment = "browser" | "server";
export type BenchmarkBackendId = string;
export type BenchmarkResultStatus = "completed" | "skipped" | "failed";

export interface BenchmarkAdapterInputMatrix {
  readonly rows: number;
  readonly columns: number;
  readonly data: Float32Array | Float64Array;
}

export interface BenchmarkAdapterCscMatrix {
  readonly rows: number;
  readonly columns: number;
  readonly values: Float32Array | Float64Array;
  readonly rowIndices: Int32Array;
  readonly columnPointers: Int32Array;
}

export interface BenchmarkAdapterCsrMatrix {
  readonly rows: number;
  readonly columns: number;
  readonly values: Float32Array | Float64Array;
  readonly columnIndices: Int32Array;
  readonly rowPointers: Int32Array;
}

export interface BenchmarkAdapterEllpackMatrix {
  readonly rows: number;
  readonly columns: number;
  readonly values: Float32Array | Float64Array;
  readonly columnIndices: Int32Array;
  readonly maxNonZerosPerRow: number;
  readonly nonZeroCount: number;
}

export interface BenchmarkAdapterInputs {
  readonly vector: Float32Array | Float64Array;
  readonly otherVector?: Float32Array | Float64Array;
  readonly leftMatrix?: BenchmarkAdapterInputMatrix;
  readonly rightMatrix?: BenchmarkAdapterInputMatrix;
  readonly leftCsr?: BenchmarkAdapterCsrMatrix;
  readonly leftCsc?: BenchmarkAdapterCscMatrix;
  readonly leftEllpack?: BenchmarkAdapterEllpackMatrix;
}

export interface BenchmarkAdapterPrepareRequest {
  readonly testCase: BenchmarkCase;
  readonly inputs: BenchmarkAdapterInputs;
}

export interface PreparedBenchmarkAdapterCase {
  readonly execute: () => number;
  readonly densityActual: number;
  readonly nnz: number;
  readonly ellpackStorageSlots: number;
  readonly ellpackPaddingRatio: number;
  readonly estimatedMemoryBytes: number;
}

export interface UploadedBenchmarkAdapterModule {
  readonly metadata: Omit<BenchmarkBackendDefinition, "id"> & { readonly id: string };
  readonly createAdapter: () => {
    readonly prepare: (request: BenchmarkAdapterPrepareRequest) => PreparedBenchmarkAdapterCase | null | Promise<PreparedBenchmarkAdapterCase | null>;
  };
}

export interface UploadedBenchmarkAdapterDefinition {
  readonly id: string;
  readonly name: string;
  readonly version: string;
  readonly environment: BenchmarkEnvironment;
  readonly description: string;
  readonly operations: readonly OperationName[];
  readonly matrixFormats: readonly MatrixFormat[];
  readonly source: string;
}

export interface BenchmarkBackendDefinition {
  readonly id: BenchmarkBackendId;
  readonly name: string;
  readonly version: string;
  readonly environment: BenchmarkEnvironment;
  readonly description: string;
  readonly operations: readonly OperationName[];
  readonly matrixFormats: readonly MatrixFormat[];
  readonly available?: boolean;
}

export const BENCHMARK_BACKENDS: readonly BenchmarkBackendDefinition[] = [
  {
    id: "trace-core",
    name: "TRACE Matrix Core",
    version: "0.5.0",
    environment: "browser",
    description: "Implementación propia TypeScript del core numérico.",
    operations: OPERATION_CATALOG.filter((spec) => spec.status === "implemented").map((spec) => spec.id),
    matrixFormats: ["dense", "csr", "csc", "ellpack"],
  },
  {
    id: "trace-rust-wasm",
    name: "TRACE Rust WASM",
    version: "0.5.0",
    environment: "browser",
    available: false,
    description: "Core Rust propio compilado a WebAssembly; habilitado tras construir el paquete wasm.",
    operations: OPERATION_CATALOG.filter((spec) => spec.status === "implemented").map((spec) => spec.id),
    matrixFormats: ["dense", "csr", "csc", "ellpack"],
  },
  {
    id: "javascript-baseline",
    name: "JavaScript baseline",
    version: "native-browser",
    environment: "browser",
    description: "Kernel de referencia escrito directamente con TypedArrays y JavaScript.",
    operations: ["vectorAdd", "vectorScale", "dotProduct", "vectorNorm2", "matrixVectorMultiply", "matrixMatrixMultiply"],
    matrixFormats: ["dense"],
  },
  {
    id: "ml-matrix",
    name: "ml-matrix",
    version: "6.15.0",
    environment: "browser",
    description: "Librería matricial JavaScript de uso real; se compara solo donde su API cubre el caso.",
    operations: ["vectorAdd", "vectorScale", "dotProduct", "vectorNorm2", "matrixVectorMultiply", "matrixMatrixMultiply"],
    matrixFormats: ["dense"],
  },
  {
    id: "ndarray-blas-gemv",
    name: "ndarray + BLAS GEMV",
    version: "ndarray 1.0.18 · ndarray-ops 1.2.2 · gemv 1.0.4",
    environment: "browser",
    description: "Ecosistema scijs basado en ndarrays y un kernel BLAS Level 2 GEMV.",
    operations: ["matrixVectorMultiply", "vectorNorm2"],
    matrixFormats: ["dense"],
  },
];

export const NATIVE_BENCHMARK_BACKENDS: readonly BenchmarkBackendDefinition[] = [
  { id: "trace-rust-native-cpu", name: "TRACE Rust Native CPU", version: "0.5.0", environment: "server", available: true, description: "Core Rust nativo con seleccion automatica de SIMD, cache blocking y paralelismo CPU.", operations: OPERATION_CATALOG.filter((spec) => spec.status === "implemented").map((spec) => spec.id), matrixFormats: ["dense", "csr", "csc", "ellpack"] },
  { id: "trace-rust-native-gpu", available: false, name: "TRACE Rust Native GPU", version: "0.5.0", environment: "server", description: "Backend GPU de TRACE. La disponibilidad depende del adaptador GPU detectado por el servidor.", operations: OPERATION_CATALOG.filter((spec) => spec.status === "implemented").map((spec) => spec.id), matrixFormats: ["dense", "csr", "csc", "ellpack"] },
  { id: "openblas", available: false, name: "OpenBLAS", version: "native adapter", environment: "server", description: "Adapter nativo de referencia BLAS/LAPACK; requiere enlazado en el servidor.", operations: ["dotProduct", "matrixVectorMultiply", "matrixMatrixMultiply"], matrixFormats: ["dense"] },
  { id: "blis", available: false, name: "BLIS", version: "native adapter", environment: "server", description: "Framework BLAS con microkernels y paralelismo multinivel.", operations: ["dotProduct", "matrixVectorMultiply", "matrixMatrixMultiply"], matrixFormats: ["dense"] },
  { id: "onemkl", available: false, name: "Intel oneMKL", version: "native adapter", environment: "server", description: "Backend Intel oneMKL para CPU y, cuando el adapter lo habilite, GPU/SYCL.", operations: ["dotProduct", "matrixVectorMultiply", "matrixMatrixMultiply"], matrixFormats: ["dense"] },
  { id: "aocl", available: false, name: "AMD AOCL", version: "native adapter", environment: "server", description: "Backend AMD optimizado para plataformas compatibles.", operations: ["dotProduct", "matrixVectorMultiply", "matrixMatrixMultiply"], matrixFormats: ["dense"] },
  { id: "eigen", available: false, name: "Eigen", version: "native adapter", environment: "server", description: "Libreria C++ de algebra lineal para comparacion nativa.", operations: ["dotProduct", "matrixVectorMultiply", "matrixMatrixMultiply"], matrixFormats: ["dense"] },
  { id: "armpl", available: false, name: "Arm Performance Libraries", version: "native adapter", environment: "server", description: "Backend para plataformas ARM cuando esta instalado.", operations: ["dotProduct", "matrixVectorMultiply", "matrixMatrixMultiply"], matrixFormats: ["dense"] },
  { id: "accelerate", available: false, name: "Apple Accelerate", version: "native adapter", environment: "server", description: "Backend Apple para macOS/Apple Silicon cuando esta disponible.", operations: ["dotProduct", "matrixVectorMultiply", "matrixMatrixMultiply"], matrixFormats: ["dense"] },
  { id: "cublas", available: false, name: "NVIDIA cuBLAS", version: "native adapter", environment: "server", description: "Backend NVIDIA CUDA para operaciones BLAS sobre GPU.", operations: ["dotProduct", "matrixVectorMultiply", "matrixMatrixMultiply"], matrixFormats: ["dense"] },
  { id: "rocblas", available: false, name: "AMD rocBLAS", version: "native adapter", environment: "server", description: "Backend AMD ROCm para operaciones BLAS sobre GPU.", operations: ["dotProduct", "matrixVectorMultiply", "matrixMatrixMultiply"], matrixFormats: ["dense"] },
  { id: "nvpl", available: false, name: "NVIDIA Performance Libraries", version: "native adapter", environment: "server", description: "Backend NVIDIA HPC para plataformas compatibles.", operations: ["dotProduct", "matrixVectorMultiply", "matrixMatrixMultiply"], matrixFormats: ["dense"] },
];

export interface BenchmarkConfiguration {
  readonly operation: OperationName;
  readonly matrixFormats: MatrixFormat[];
  readonly scalarTypes: ScalarType[];
  readonly sizes: number[];
  readonly densities: number[];
  readonly warmupIterations: number;
  readonly measuredIterations: number;
  readonly casesPerConfiguration: number;
  readonly targetSampleMilliseconds: number;
  readonly selectedBackends: BenchmarkBackendId[];
}

export interface BenchmarkCase {
  readonly id: string;
  readonly operation: OperationName;
  readonly matrixFormat: MatrixFormat | "none";
  readonly scalarType: ScalarType;
  readonly size: number;
  readonly rows: number;
  readonly columns: number;
  readonly vectorLength: number;
  readonly densityRequested: number;
  readonly seed: number;
}

export interface BenchmarkStatistics {
  readonly min: number;
  readonly p50: number;
  readonly p90: number;
  readonly p95: number;
  readonly p99: number;
  readonly max: number;
  readonly mean: number;
  readonly standardDeviation: number;
  readonly coefficientOfVariation: number;
  readonly iqr: number;
  readonly confidenceInterval95: number;
}

export interface BenchmarkResult {
  readonly id: string;
  readonly runId: string;
  readonly caseId: string;
  readonly timestamp: string;
  readonly operation: OperationName;
  readonly matrixFormat: MatrixFormat | "none";
  readonly scalarType: ScalarType;
  readonly rows: number;
  readonly columns: number;
  readonly vectorLength: number;
  readonly densityRequested: number;
  readonly densityActual: number;
  readonly nnz: number;
  readonly ellpackStorageSlots: number;
  readonly ellpackPaddingRatio: number;
  readonly warmupIterations: number;
  readonly measuredIterations: number;
  readonly innerIterations: number;
  readonly statistics: BenchmarkStatistics;
  readonly estimatedFlops: number;
  readonly estimatedGflops: number;
  readonly estimatedMemoryBytes: number;
  readonly backendId: BenchmarkBackendId;
  readonly backendName: string;
  readonly backendVersion: string;
  readonly environment: BenchmarkEnvironment;
  readonly status: BenchmarkResultStatus;
  readonly skipReason?: string;
  readonly errorMessage?: string;
  readonly checksum: number;
}

export interface BenchmarkRun {
  readonly id: string;
  readonly startedAt: string;
  readonly completedAt: string;
  readonly label: string;
  readonly configuration: BenchmarkConfiguration;
  readonly selectedBackends: BenchmarkBackendId[];
  readonly environment: EnvironmentSnapshot;
  readonly caseCount: number;
  readonly resultCount: number;
}

export interface EnvironmentSnapshot {
  readonly userAgent: string;
  readonly platform: string;
  readonly language: string;
  readonly hardwareConcurrency: number | null;
  readonly deviceMemoryGb: number | null;
  readonly crossOriginIsolated: boolean;
  readonly sharedArrayBuffer: boolean;
  readonly timestamp: string;
  readonly world: BenchmarkEnvironment;
}

export interface BenchmarkSnapshot {
  readonly run: BenchmarkRun;
  readonly results: BenchmarkResult[];
}

export interface ServerBenchmarkOptions {
  readonly serverUrl: string;
  readonly execution: "cpu" | "gpu";
}

interface ServerExecuteResponse {
  readonly operation: string;
  readonly execution: string;
  readonly backend: string;
  readonly elapsed_ns: number;
  readonly checksum: number;
  readonly values: number[];
}


interface PreparedCase {
  readonly execute: () => number;
  readonly densityActual: number;
  readonly nnz: number;
  readonly ellpackStorageSlots: number;
  readonly ellpackPaddingRatio: number;
  readonly estimatedMemoryBytes: number;
}

const NUMBER_EPSILON = 1e-12;

/**
 * Obtiene la información del entorno sin asumir APIs no disponibles en el
 * navegador. Estos datos quedan ligados a cada ejecución para reproducibilidad.
 */
export function readEnvironmentSnapshot(): EnvironmentSnapshot {
  const navigatorWithMemory = navigator as Navigator & { deviceMemory?: number };
  const globalWithIsolation = globalThis as typeof globalThis & { crossOriginIsolated?: boolean };

  return {
    userAgent: navigator.userAgent,
    platform: navigator.platform,
    language: navigator.language,
    hardwareConcurrency: navigator.hardwareConcurrency ?? null,
    deviceMemoryGb: navigatorWithMemory.deviceMemory ?? null,
    crossOriginIsolated: globalWithIsolation.crossOriginIsolated ?? false,
    sharedArrayBuffer: typeof SharedArrayBuffer !== "undefined",
    timestamp: new Date().toISOString(),
    world: "browser",
  };
}

/**
 * Genera el banco cartesiano solicitado por el usuario.
 */
function operationSpec(operation: OperationName) {
  return OPERATION_CATALOG.find((spec) => spec.id === operation);
}

/**
 * Determina si una operación necesita una representación matricial.
 */
function isMatrixOperation(operation: OperationName): boolean {
  return operationSpec(operation)?.category !== "level1";
}

/**
 * Determina los formatos válidos para una operación concreta.
 *
 * Las factorizaciones y operaciones algebraicas densas no se benchmarkean
 * sobre CSR/ELLPACK hasta disponer de kernels matemáticamente equivalentes.
 */
function formatsForOperation(configuration: BenchmarkConfiguration): MatrixFormat[] | ["none"] {
  if (!isMatrixOperation(configuration.operation)) return ["none"];
  if (configuration.operation === "matrixVectorMultiply") return configuration.matrixFormats;
  return ["dense"];
}

export function generateBenchmarkCases(configuration: BenchmarkConfiguration): BenchmarkCase[] {
  const cases: BenchmarkCase[] = [];
  let caseNumber = 0;

  for (const scalarType of configuration.scalarTypes) {
    for (const size of configuration.sizes) {
      const densities = isMatrixOperation(configuration.operation) ? configuration.densities : [1];
      for (const densityRequested of densities) {
        const formats = formatsForOperation(configuration);
        const effectiveSize = configuration.operation === "crossProduct" ? 3 : size;
        const effectiveDensity = isMatrixOperation(configuration.operation) ? densityRequested : 1;

        for (const matrixFormat of formats) {
          for (let repetition = 0; repetition < configuration.casesPerConfiguration; repetition += 1) {
            caseNumber += 1;
            cases.push({
              id: `case-${caseNumber}`,
              operation: configuration.operation,
              matrixFormat,
              scalarType,
              size: effectiveSize,
              rows: effectiveSize,
              columns: effectiveSize,
              vectorLength: effectiveSize,
              densityRequested: effectiveDensity,
              seed: 1000 + caseNumber * 7919,
            });
          }
        }
      }
    }
  }

  return cases;
}

/**
 * Crea una matriz densa reproducible y aplica la densidad solicitada antes de
 * construir cualquier representación sparse.
 */
function createRandomDenseMatrix(
  rows: number,
  columns: number,
  scalarType: ScalarType,
  density: number,
  seed: number,
): DenseMatrix {
  const data = createRandomData(rows * columns, scalarType, seed);
  const matrix = createDenseMatrix(rows, columns, scalarType, data);

  if (density >= 1) return matrix;

  let state = (seed ^ 0xa5a5a5a5) >>> 0;
  for (let index = 0; index < matrix.data.length; index += 1) {
    state = (1664525 * state + 1013904223) >>> 0;
    if (state / 0x100000000 > density) matrix.data[index] = 0;
  }

  return matrix;
}

/**
 * Genera una matriz compatible con las restricciones matemáticas del kernel.
 *
 * Las operaciones como Cholesky o una resolución triangular no admiten un
 * input aleatorio arbitrario; preparar un caso válido evita que el benchmark
 * mida errores de dominio en lugar del algoritmo.
 */
function createOperationDenseMatrix(testCase: BenchmarkCase, seed: number): DenseMatrix {
  const matrix = createRandomDenseMatrix(testCase.rows, testCase.columns, testCase.scalarType, testCase.densityRequested, seed);
  if (testCase.rows !== testCase.columns) return matrix;

  const data = matrix.data;
  if (["cholesky", "ldlt", "symmetricEigen"].includes(testCase.operation)) {
    for (let row = 0; row < matrix.rows; row += 1) {
      for (let column = row + 1; column < matrix.columns; column += 1) {
        const value = (data[row * matrix.columns + column] + data[column * matrix.columns + row]) * 0.5;
        data[row * matrix.columns + column] = value;
        data[column * matrix.columns + row] = value;
      }
    }
    if (testCase.operation === "cholesky") {
      for (let row = 0; row < matrix.rows; row += 1) {
        let offDiagonal = 0;
        for (let column = 0; column < matrix.columns; column += 1) {
          if (column !== row) offDiagonal += Math.abs(data[row * matrix.columns + column]);
        }
        data[row * matrix.columns + row] = offDiagonal + 1;
      }
    }
  }

  if (["triangularSolve", "trsm"].includes(testCase.operation)) {
    for (let row = 0; row < matrix.rows; row += 1) {
      for (let column = row + 1; column < matrix.columns; column += 1) data[row * matrix.columns + column] = 0;
      data[row * matrix.columns + row] = Math.abs(data[row * matrix.columns + row]) + 1;
    }
  }

  return matrix;
}

/**
 * Calcula NNZ y densidad real sobre el buffer canónico.
 */
function inspectDenseMatrix(matrix: DenseMatrix): { nnz: number; density: number } {
  let nnz = 0;
  for (const value of matrix.data) {
    if (Math.abs(value) > NUMBER_EPSILON) nnz += 1;
  }

  const total = matrix.rows * matrix.columns;
  return { nnz, density: total === 0 ? 0 : nnz / total };
}

/**
 * Construye los mismos operandos canónicos para adaptadores externos.
 *
 * El contrato de plugins recibe datos ya generados con la semilla del caso.
 * Así una librería subida no puede cambiar silenciosamente el workload para
 * favorecer su propia implementación.
 */
function createAdapterInputs(testCase: BenchmarkCase): BenchmarkAdapterInputs {
  const vector = createRandomData(testCase.vectorLength, testCase.scalarType, testCase.seed + 1);
  const otherVector = ["vectorAdd", "axpy", "crossProduct", "dotProduct"].includes(testCase.operation)
    ? createRandomData(testCase.vectorLength, testCase.scalarType, testCase.seed + 2)
    : undefined;

  if (!isMatrixOperation(testCase.operation)) {
    return { vector, otherVector };
  }

  const leftDense = createOperationDenseMatrix(testCase, testCase.seed + 3);
  const leftCsr = denseToCsr(leftDense);
  const leftCsc = denseToCsc(leftDense);
  const leftEllpack = denseToEllpack(leftDense);
  const rightDense = ["matrixMatrixMultiply", "solveMultipleRhs", "trsm", "syrk"].includes(testCase.operation)
    ? createRandomDenseMatrix(testCase.columns, testCase.columns, testCase.scalarType, testCase.densityRequested, testCase.seed + 4)
    : undefined;

  return {
    vector,
    otherVector,
    leftMatrix: { rows: leftDense.rows, columns: leftDense.columns, data: leftDense.data },
    rightMatrix: rightDense ? { rows: rightDense.rows, columns: rightDense.columns, data: rightDense.data } : undefined,
    leftCsr: {
      rows: leftCsr.rows,
      columns: leftCsr.columns,
      values: leftCsr.values,
      columnIndices: leftCsr.columnIndices,
      rowPointers: leftCsr.rowPointers,
    },
    leftCsc: {
      rows: leftCsc.rows,
      columns: leftCsc.columns,
      values: leftCsc.values,
      rowIndices: leftCsc.rowIndices,
      columnPointers: leftCsc.columnPointers,
    },
    leftEllpack: {
      rows: leftEllpack.rows,
      columns: leftEllpack.columns,
      values: leftEllpack.values,
      columnIndices: leftEllpack.columnIndices,
      maxNonZerosPerRow: leftEllpack.maxNonZerosPerRow,
      nonZeroCount: leftEllpack.nonZeroCount,
    },
  };
}

/**
 * Construye los operandos del core propio fuera de la región cronometrada.
 */
function prepareTraceCase(testCase: BenchmarkCase): PreparedCase {
  const scalarBytes = testCase.scalarType === "float32" ? 4 : 8;
  const vector = createVector(createRandomData(testCase.vectorLength, testCase.scalarType, testCase.seed + 1), testCase.scalarType);

  if (testCase.operation === "vectorAdd") {
    const other = createVector(createRandomData(testCase.vectorLength, testCase.scalarType, testCase.seed + 2), testCase.scalarType);
    return {
      execute: () => vectorAdd(vector, other).data[0],
      densityActual: 1,
      nnz: 0,
      ellpackStorageSlots: 0,
      ellpackPaddingRatio: 0,
      estimatedMemoryBytes: vector.length * scalarBytes * 2,
    };
  }

  if (testCase.operation === "vectorScale") {
    return {
      execute: () => vectorScale(vector, 1.25).data[0],
      densityActual: 1,
      nnz: 0,
      ellpackStorageSlots: 0,
      ellpackPaddingRatio: 0,
      estimatedMemoryBytes: vector.length * scalarBytes,
    };
  }

  if (testCase.operation === "dotProduct") {
    return {
      execute: () => dotProduct(vector, vector),
      densityActual: 1,
      nnz: 0,
      ellpackStorageSlots: 0,
      ellpackPaddingRatio: 0,
      estimatedMemoryBytes: vector.length * scalarBytes * 2,
    };
  }

  if (testCase.operation === "vectorNorm2") {
    return {
      execute: () => vectorNorm2(vector),
      densityActual: 1,
      nnz: 0,
      ellpackStorageSlots: 0,
      ellpackPaddingRatio: 0,
      estimatedMemoryBytes: vector.length * scalarBytes,
    };
  }

  const dense = createRandomDenseMatrix(testCase.rows, testCase.columns, testCase.scalarType, testCase.densityRequested, testCase.seed + 3);
  const inspection = inspectDenseMatrix(dense);

  if (testCase.operation === "matrixVectorMultiply") {
    if (testCase.matrixFormat === "csr") {
      const matrix = denseToCsr(dense);
      return {
        execute: () => matrixVectorMultiply(matrix, vector).data[0],
        densityActual: inspection.density,
        nnz: inspection.nnz,
        ellpackStorageSlots: 0,
        ellpackPaddingRatio: 0,
        estimatedMemoryBytes: matrix.values.byteLength + matrix.columnIndices.byteLength + matrix.rowPointers.byteLength + vector.data.byteLength,
      };
    }

    if (testCase.matrixFormat === "ellpack") {
      const matrix = denseToEllpack(dense);
      const ellpackStorageSlots = matrix.maxNonZerosPerRow * matrix.rows;
      return {
        execute: () => matrixVectorMultiply(matrix, vector).data[0],
        densityActual: inspection.density,
        nnz: inspection.nnz,
        ellpackStorageSlots,
        ellpackPaddingRatio: ellpackStorageSlots === 0 ? 0 : 1 - inspection.nnz / ellpackStorageSlots,
        estimatedMemoryBytes: matrix.values.byteLength + matrix.columnIndices.byteLength + vector.data.byteLength,
      };
    }

    return {
      execute: () => matrixVectorMultiply(dense, vector).data[0],
      densityActual: inspection.density,
      nnz: inspection.nnz,
      ellpackStorageSlots: 0,
      ellpackPaddingRatio: 0,
      estimatedMemoryBytes: dense.data.byteLength + vector.data.byteLength,
    };
  }

  const rightDense = createRandomDenseMatrix(testCase.columns, testCase.columns, testCase.scalarType, testCase.densityRequested, testCase.seed + 4);
  return {
    execute: () => matrixMatrixMultiply(dense, rightDense).data[0],
    densityActual: inspection.density,
    nnz: inspection.nnz,
    ellpackStorageSlots: 0,
    ellpackPaddingRatio: 0,
    estimatedMemoryBytes: dense.data.byteLength + rightDense.data.byteLength,
  };
}

/**
 * Implementa directamente los kernels dense con TypedArrays para disponer de
 * una referencia JavaScript independiente del core propio.
 */
function prepareJavaScriptCase(testCase: BenchmarkCase): PreparedCase | null {
  if (testCase.matrixFormat !== "none" && testCase.matrixFormat !== "dense") return null;

  const scalarBytes = testCase.scalarType === "float32" ? 4 : 8;
  const VectorArray = testCase.scalarType === "float32" ? Float32Array : Float64Array;
  const vector = new VectorArray(createRandomData(testCase.vectorLength, testCase.scalarType, testCase.seed + 1));

  if (testCase.operation === "vectorAdd") {
    const other = new VectorArray(createRandomData(testCase.vectorLength, testCase.scalarType, testCase.seed + 2));
    return {
      execute: () => {
        let first = 0;
        for (let index = 0; index < vector.length; index += 1) {
          const value = vector[index] + other[index];
          if (index === 0) first = value;
        }
        return first;
      },
      densityActual: 1,
      nnz: 0,
      ellpackStorageSlots: 0,
      ellpackPaddingRatio: 0,
      estimatedMemoryBytes: vector.byteLength + other.byteLength,
    };
  }

  if (testCase.operation === "vectorScale") {
    return {
      execute: () => {
        let first = 0;
        for (let index = 0; index < vector.length; index += 1) {
          const value = vector[index] * 1.25;
          if (index === 0) first = value;
        }
        return first;
      },
      densityActual: 1,
      nnz: 0,
      ellpackStorageSlots: 0,
      ellpackPaddingRatio: 0,
      estimatedMemoryBytes: vector.byteLength,
    };
  }

  if (testCase.operation === "dotProduct") {
    return {
      execute: () => {
        let sum = 0;
        for (let index = 0; index < vector.length; index += 1) sum += vector[index] * vector[index];
        return sum;
      },
      densityActual: 1,
      nnz: 0,
      ellpackStorageSlots: 0,
      ellpackPaddingRatio: 0,
      estimatedMemoryBytes: vector.byteLength,
    };
  }

  if (testCase.operation === "vectorNorm2") {
    return {
      execute: () => {
        let sum = 0;
        for (let index = 0; index < vector.length; index += 1) sum += vector[index] * vector[index];
        return Math.sqrt(sum);
      },
      densityActual: 1,
      nnz: 0,
      ellpackStorageSlots: 0,
      ellpackPaddingRatio: 0,
      estimatedMemoryBytes: vector.byteLength,
    };
  }

  const dense = createRandomDenseMatrix(testCase.rows, testCase.columns, testCase.scalarType, testCase.densityRequested, testCase.seed + 3);
  const inspection = inspectDenseMatrix(dense);

  if (testCase.operation === "matrixVectorMultiply") {
    return {
      execute: () => {
        let first = 0;
        for (let row = 0; row < dense.rows; row += 1) {
          let sum = 0;
          const offset = row * dense.columns;
          for (let column = 0; column < dense.columns; column += 1) sum += dense.data[offset + column] * vector[column];
          if (row === 0) first = sum;
        }
        return first;
      },
      densityActual: inspection.density,
      nnz: inspection.nnz,
      ellpackStorageSlots: 0,
      ellpackPaddingRatio: 0,
      estimatedMemoryBytes: dense.data.byteLength + vector.byteLength,
    };
  }

  const right = createRandomDenseMatrix(testCase.columns, testCase.columns, testCase.scalarType, testCase.densityRequested, testCase.seed + 4);
  return {
    execute: () => {
      let first = 0;
      for (let row = 0; row < dense.rows; row += 1) {
        for (let column = 0; column < right.columns; column += 1) {
          let sum = 0;
          for (let inner = 0; inner < dense.columns; inner += 1) {
            sum += dense.data[row * dense.columns + inner] * right.data[inner * right.columns + column];
          }
          if (row === 0 && column === 0) first = sum;
        }
      }
      return first;
    },
    densityActual: inspection.density,
    nnz: inspection.nnz,
    ellpackStorageSlots: 0,
    ellpackPaddingRatio: 0,
    estimatedMemoryBytes: dense.data.byteLength + right.data.byteLength,
  };
}

/**
 * Adaptador de ml-matrix. La construcción de sus objetos ocurre fuera del
 * timing; la región medida contiene únicamente la operación de la librería.
 */
function prepareMlMatrixCase(testCase: BenchmarkCase): PreparedCase | null {
  if (testCase.matrixFormat !== "none" && testCase.matrixFormat !== "dense") return null;

  const scalarBytes = testCase.scalarType === "float32" ? 4 : 8;
  const vectorValues = Array.from(createRandomData(testCase.vectorLength, testCase.scalarType, testCase.seed + 1));

  if (testCase.operation === "vectorAdd") {
    const left = MlMatrix.columnVector(vectorValues);
    const right = MlMatrix.columnVector(Array.from(createRandomData(testCase.vectorLength, testCase.scalarType, testCase.seed + 2)));
    return {
      execute: () => MlMatrix.add(left, right).get(0, 0),
      densityActual: 1,
      nnz: 0,
      ellpackStorageSlots: 0,
      ellpackPaddingRatio: 0,
      estimatedMemoryBytes: vectorValues.length * scalarBytes * 2,
    };
  }

  if (testCase.operation === "vectorScale") {
    const vector = MlMatrix.columnVector(vectorValues);
    return {
      execute: () => MlMatrix.mul(vector, 1.25).get(0, 0),
      densityActual: 1,
      nnz: 0,
      ellpackStorageSlots: 0,
      ellpackPaddingRatio: 0,
      estimatedMemoryBytes: vectorValues.length * scalarBytes,
    };
  }

  if (testCase.operation === "dotProduct") {
    const vector = MlMatrix.columnVector(vectorValues);
    const transpose = vector.transpose();
    return {
      execute: () => transpose.mmul(vector).get(0, 0),
      densityActual: 1,
      nnz: 0,
      ellpackStorageSlots: 0,
      ellpackPaddingRatio: 0,
      estimatedMemoryBytes: vectorValues.length * scalarBytes,
    };
  }

  if (testCase.operation === "vectorNorm2") {
    const vector = MlMatrix.columnVector(vectorValues);
    return {
      execute: () => Math.sqrt(vector.transpose().mmul(vector).get(0, 0)),
      densityActual: 1,
      nnz: 0,
      ellpackStorageSlots: 0,
      ellpackPaddingRatio: 0,
      estimatedMemoryBytes: vectorValues.length * scalarBytes,
    };
  }

  const dense = createRandomDenseMatrix(testCase.rows, testCase.columns, testCase.scalarType, testCase.densityRequested, testCase.seed + 3);
  const inspection = inspectDenseMatrix(dense);
  const left = new MlMatrix(Array.from({ length: dense.rows }, (_, row) => Array.from(dense.data.slice(row * dense.columns, (row + 1) * dense.columns))));

  if (testCase.operation === "matrixVectorMultiply") {
    const vector = MlMatrix.columnVector(vectorValues);
    return {
      execute: () => left.mmul(vector).get(0, 0),
      densityActual: inspection.density,
      nnz: inspection.nnz,
      ellpackStorageSlots: 0,
      ellpackPaddingRatio: 0,
      estimatedMemoryBytes: dense.data.byteLength + vectorValues.length * scalarBytes,
    };
  }

  const rightDense = createRandomDenseMatrix(testCase.columns, testCase.columns, testCase.scalarType, testCase.densityRequested, testCase.seed + 4);
  const right = new MlMatrix(Array.from({ length: rightDense.rows }, (_, row) => Array.from(rightDense.data.slice(row * rightDense.columns, (row + 1) * rightDense.columns))));

  return {
    execute: () => left.mmul(right).get(0, 0),
    densityActual: inspection.density,
    nnz: inspection.nnz,
    ellpackStorageSlots: 0,
    ellpackPaddingRatio: 0,
    estimatedMemoryBytes: dense.data.byteLength + rightDense.data.byteLength,
  };
}


/**
 * Adaptador del ecosistema scijs ndarray. Su cobertura deliberadamente se
 * limita a las operaciones que tienen un kernel de biblioteca explícito y
 * comparable, especialmente BLAS GEMV.
 */
function prepareNdarrayCase(testCase: BenchmarkCase): PreparedCase | null {
  if (testCase.matrixFormat !== "none" && testCase.matrixFormat !== "dense") return null;
  if (testCase.operation !== "matrixVectorMultiply" && testCase.operation !== "vectorNorm2") return null;

  const scalarBytes = testCase.scalarType === "float32" ? 4 : 8;
  const vectorData = createRandomData(testCase.vectorLength, testCase.scalarType, testCase.seed + 1);
  const vector = ndarray(vectorData, [testCase.vectorLength]);

  if (testCase.operation === "vectorNorm2") {
    return {
      execute: () => ndarrayOps.norm2(vector),
      densityActual: 1,
      nnz: 0,
      ellpackStorageSlots: 0,
      ellpackPaddingRatio: 0,
      estimatedMemoryBytes: vectorData.byteLength,
    };
  }

  const dense = createRandomDenseMatrix(testCase.rows, testCase.columns, testCase.scalarType, testCase.densityRequested, testCase.seed + 3);
  const inspection = inspectDenseMatrix(dense);
  const matrix = ndarray(dense.data, [dense.rows, dense.columns]);
  const outputData = testCase.scalarType === "float32" ? new Float32Array(testCase.rows) : new Float64Array(testCase.rows);
  const output = ndarray(outputData, [testCase.rows]);

  return {
    execute: () => {
      gemv(1, matrix, vector, 0, output);
      return outputData[0];
    },
    densityActual: inspection.density,
    nnz: inspection.nnz,
    ellpackStorageSlots: 0,
    ellpackPaddingRatio: 0,
    estimatedMemoryBytes: dense.data.byteLength + vectorData.byteLength + outputData.byteLength,
  };
}

/**
 * Devuelve el adaptador adecuado para un backend y evita mezclar operaciones
 * no soportadas con resultados numéricos que aparentarían ser comparables.
 */
async function prepareUploadedBackendCase(
  definition: UploadedBenchmarkAdapterDefinition,
  testCase: BenchmarkCase,
): Promise<PreparedCase | null> {
  const moduleUrl = URL.createObjectURL(new Blob([definition.source], { type: "text/javascript" }));
  try {
    const module = await import(/* @vite-ignore */ moduleUrl) as UploadedBenchmarkAdapterModule;
    if (!module?.metadata || typeof module.createAdapter !== "function") {
      throw new Error("El adaptador no exporta metadata y createAdapter().");
    }

    const adapter = module.createAdapter();
    const prepared = await adapter.prepare({ testCase, inputs: createAdapterInputs(testCase) });
    return prepared;
  } finally {
    URL.revokeObjectURL(moduleUrl);
  }
}

/**
 * Carga el módulo wasm-bindgen generado por wasm-pack.
 *
 * La ruta es absoluta para que Vite pueda servir el artefacto compilado sin
 * empaquetarlo en el bundle principal. Si todavía no existe, el backend queda
 * correctamente marcado como no disponible y no se inventan mediciones.
 */
let rustWasmModulePromise: Promise<any> | null = null;

async function loadRustWasmModule(): Promise<any> {
  if (!rustWasmModulePromise) {
    rustWasmModulePromise = import(/* @vite-ignore */ "/wasm/trace_numerical_wasm.js")
      .then(async (module) => {
        if (typeof module.default === "function") await module.default("/wasm/trace_numerical_wasm_bg.wasm");
        return module;
      });
  }
  return rustWasmModulePromise;
}

/**
 * Prepara un caso para la implementación Rust/WASM propia.
 *
 * La preparación de TypedArrays queda fuera del tramo cronometrado. El kernel
 * medido es exclusivamente la llamada a la función exportada por WASM.
 */
async function prepareRustWasmCase(testCase: BenchmarkCase): Promise<PreparedCase | null> {
  if (testCase.scalarType !== "float64") return null;

  const wasm = await loadRustWasmModule();
  const vector = new Float64Array(createRandomData(testCase.vectorLength, "float64", testCase.seed + 1));
  const otherVector = new Float64Array(createRandomData(testCase.vectorLength, "float64", testCase.seed + 2));
  const dense = isMatrixOperation(testCase.operation) ? createOperationDenseMatrix(testCase, testCase.seed + 3) : null;
  const right = isMatrixOperation(testCase.operation) ? createRandomDenseMatrix(testCase.columns, testCase.columns, "float64", testCase.densityRequested, testCase.seed + 4) : null;
  const inspection = dense ? inspectDenseMatrix(dense) : { nnz: 0, density: 1 };
  const csr = dense ? denseToCsr(dense) : null;
  const csc = dense ? denseToCsc(dense) : null;
  const ellpack = dense ? denseToEllpack(dense) : null;
  const scalarBytes = 8;

  const executeOperation = (): number => {
    switch (testCase.operation) {
      case "vectorAdd": return wasm.vector_add_wasm(vector, otherVector)[0];
      case "vectorScale": return wasm.vector_scale_wasm(vector, 1.25)[0];
      case "axpy": return wasm.axpy_wasm(1.25, vector, otherVector)[0];
      case "crossProduct": return wasm.cross_product_wasm(vector.slice(0, 3), otherVector.slice(0, 3))[0];
      case "dotProduct": return wasm.dot_product_wasm(vector, otherVector);
      case "vectorNorm2": return wasm.vector_norm2_wasm(vector);
      case "gemv": return wasm.gemv_wasm(dense!.rows, dense!.columns, dense!.data, 1, vector, 0, new Float64Array(dense!.rows))[0];
      case "matrixVectorMultiply":
        if (testCase.matrixFormat === "csr") return wasm.csr_matrix_vector_multiply_wasm(csr!.rows, csr!.columns, csr!.values, csr!.columnIndices, csr!.rowPointers, vector)[0];
        if (testCase.matrixFormat === "csc") return wasm.csc_matrix_vector_multiply_wasm(csc!.rows, csc!.columns, csc!.values, csc!.rowIndices, csc!.columnPointers, vector)[0];
        if (testCase.matrixFormat === "ellpack") return wasm.ellpack_matrix_vector_multiply_wasm(ellpack!.rows, ellpack!.columns, ellpack!.values, ellpack!.columnIndices, ellpack!.maxNonZerosPerRow, ellpack!.nonZeroCount, vector)[0];
        return wasm.matrix_vector_multiply_wasm(dense!.rows, dense!.columns, dense!.data, vector)[0];
      case "matrixMatrixMultiply": return wasm.matrix_matrix_multiply_wasm(dense!.rows, dense!.columns, dense!.data, right!.rows, right!.columns, right!.data)[0];
      case "ger": return wasm.ger_wasm(dense!.rows, dense!.columns, dense!.data, 1, vector, otherVector)[0];
      case "syrk": return wasm.syrk_wasm(dense!.rows, dense!.columns, dense!.data, right!.rows, right!.data, 1, 1)[0];
      case "transpose": return wasm.transpose_wasm(dense!.rows, dense!.columns, dense!.data)[0];
      case "trace": return wasm.trace_wasm(dense!.rows, dense!.data);
      case "matrixNormFrobenius": return wasm.matrix_norm_frobenius_wasm(dense!.rows, dense!.columns, dense!.data);
      case "matrixNormOne": return wasm.matrix_norm_one_wasm(dense!.rows, dense!.columns, dense!.data);
      case "matrixNormInf": return wasm.matrix_norm_inf_wasm(dense!.rows, dense!.columns, dense!.data);
      case "matrixNormTwo": return wasm.matrix_norm_two_wasm(dense!.rows, dense!.columns, dense!.data);
      case "kroneckerProduct": return wasm.kronecker_product_wasm(dense!.rows, dense!.columns, dense!.data, 2, 2, new Float64Array([1, 0, 0, 1]))[0];
      case "determinant": return wasm.determinant_wasm(dense!.rows, dense!.data);
      case "inverse": return wasm.inverse_wasm(dense!.rows, dense!.data)[0];
      case "solveLinearSystem": return wasm.solve_linear_system_wasm(dense!.rows, dense!.data, vector)[0];
      case "solveMultipleRhs": return wasm.solve_multiple_rhs_wasm(dense!.rows, right!.columns, dense!.data, right!.data)[0];
      case "triangularSolve": return wasm.triangular_solve_wasm(dense!.rows, dense!.data, vector, true, false)[0];
      case "trsm": return wasm.trsm_wasm(dense!.rows, right!.columns, dense!.data, right!.data, true, false)[0];
      case "lu": { const result = wasm.lu_wasm(dense!.rows, dense!.data); return result.l[0] + result.u[0]; }
      case "cholesky": return wasm.cholesky_wasm(dense!.rows, dense!.data)[0];
      case "ldlt": { const result = wasm.ldlt_wasm(dense!.rows, dense!.data); return result[0][0] + result[1][0]; }
      case "qr": { const result = wasm.qr_wasm(dense!.rows, dense!.columns, dense!.data); return result.q[0] + result.r[0]; }
      case "rankRevealing": { const result = wasm.rank_revealing_qr_wasm(dense!.rows, dense!.columns, dense!.data); return result[3]; }
      case "svd": { const result = wasm.svd_wasm(dense!.rows, dense!.columns, dense!.data); return result.singular_values[0] ?? 0; }
      case "polar": { const result = wasm.polar_wasm(dense!.rows, dense!.columns, dense!.data); return result[0][0] ?? 0; }
      case "symmetricEigen": { const result = wasm.symmetric_eigen_wasm(dense!.rows, dense!.data); return result.eigenvalues[0] ?? 0; }
      case "hessenberg": { const result = wasm.hessenberg_wasm(dense!.rows, dense!.data); return result.h[0]; }
      default: return Number.NaN;
    }
  };

  return {
    execute: executeOperation,
    densityActual: inspection.density,
    nnz: inspection.nnz,
    ellpackStorageSlots: 0,
    ellpackPaddingRatio: 0,
    estimatedMemoryBytes: (dense ? (testCase.matrixFormat === "csr" ? csr!.values.byteLength + csr!.columnIndices.byteLength + csr!.rowPointers.byteLength : testCase.matrixFormat === "csc" ? csc!.values.byteLength + csc!.rowIndices.byteLength + csc!.columnPointers.byteLength : testCase.matrixFormat === "ellpack" ? ellpack!.values.byteLength + ellpack!.columnIndices.byteLength : dense.data.byteLength) + (right?.data.byteLength ?? 0) : vector.byteLength + otherVector.byteLength) + scalarBytes,
  };
}

/**
 * Devuelve el adaptador adecuado para un backend incorporado o subido.
 */
async function prepareBackendCase(
  backendId: BenchmarkBackendId,
  testCase: BenchmarkCase,
  uploadedBackends: readonly UploadedBenchmarkAdapterDefinition[],
): Promise<PreparedCase | null> {
  if (backendId === "trace-core") return prepareTraceCase(testCase);
  if (backendId === "trace-rust-wasm") return prepareRustWasmCase(testCase);
  if (backendId === "javascript-baseline") return prepareJavaScriptCase(testCase);
  if (backendId === "ml-matrix") return prepareMlMatrixCase(testCase);
  if (backendId === "ndarray-blas-gemv") return prepareNdarrayCase(testCase);

  const uploaded = uploadedBackends.find((candidate) => candidate.id === backendId);
  if (!uploaded) return null;
  return prepareUploadedBackendCase(uploaded, testCase);
}

/**
 * Calcula percentiles por interpolación lineal sobre muestras ordenadas.
 */
function percentile(sorted: number[], probability: number): number {
  if (sorted.length === 1) return sorted[0];
  const position = (sorted.length - 1) * probability;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return sorted[lower];
  const weight = position - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}

/**
 * Calcula estadísticas robustas y un intervalo de confianza aproximado del
 * 95% para la media de las muestras.
 */
function calculateStatistics(values: number[]): BenchmarkStatistics {
  const sorted = [...values].sort((left, right) => left - right);
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.length > 1
    ? values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (values.length - 1)
    : 0;
  const standardDeviation = Math.sqrt(variance);
  const p25 = percentile(sorted, 0.25);
  const p75 = percentile(sorted, 0.75);
  const standardError = values.length > 0 ? standardDeviation / Math.sqrt(values.length) : 0;

  return {
    min: sorted[0] ?? 0,
    p50: percentile(sorted, 0.5),
    p90: percentile(sorted, 0.9),
    p95: percentile(sorted, 0.95),
    p99: percentile(sorted, 0.99),
    max: sorted[sorted.length - 1] ?? 0,
    mean,
    standardDeviation,
    coefficientOfVariation: mean === 0 ? 0 : standardDeviation / mean,
    iqr: p75 - p25,
    confidenceInterval95: 1.96 * standardError,
  };
}

/**
 * Calcula FLOPs reales según la estructura efectiva del problema.
 */
function estimateFlops(testCase: BenchmarkCase, nnz: number): number {
  switch (testCase.operation) {
    case "vectorAdd":
      return testCase.vectorLength;
    case "vectorScale":
      return testCase.vectorLength;
    case "axpy":
      return 2 * testCase.vectorLength;
    case "crossProduct":
      return 9;
    case "dotProduct":
    case "vectorNorm2":
      return 2 * testCase.vectorLength;
    case "gemv":
    case "matrixVectorMultiply":
      return 2 * (testCase.matrixFormat === "dense" ? testCase.rows * testCase.columns : nnz);
    case "ger":
      return 2 * testCase.rows * testCase.columns;
    case "matrixMatrixMultiply":
    case "syrk":
      return 2 * testCase.rows * testCase.columns * testCase.columns;
    case "triangularSolve":
    case "determinant":
    case "inverse":
    case "solveLinearSystem":
    case "lu":
    case "cholesky":
    case "ldlt":
    case "bunchKaufman":
    case "polar":
    case "eigen":
    case "symmetricEigen":
    case "schur":
    case "hessenberg":
    case "jordan":
    case "frobenius":
      return (2 / 3) * testCase.size ** 3;
    case "qr":
      return 2 * testCase.rows * testCase.columns * Math.min(testCase.rows, testCase.columns);
    case "svd":
      return 4 * testCase.rows * testCase.columns * Math.min(testCase.rows, testCase.columns);
    case "trsm":
    case "solveMultipleRhs":
      return testCase.size ** 2 * testCase.columns;
    case "transpose":
    case "matrixNormFrobenius":
    case "matrixNormOne":
    case "matrixNormInf":
      return testCase.rows * testCase.columns;
    case "matrixNormTwo":
      return 60 * testCase.rows * testCase.columns;
    case "trace":
      return testCase.size;
    case "kroneckerProduct":
      return testCase.rows * testCase.columns * 4;
    case "rankRevealing":
      return 2 * testCase.rows * testCase.columns * Math.min(testCase.rows, testCase.columns);
    case "hermite":
    case "smith":
    case "popov":
      return 0;
    default:
      return 0;
  }
}

/**
 * Busca un número de iteraciones internas suficiente para que el cronómetro
 * tenga señal, sin convertir la calibración en parte de la muestra final.
 */
function calibrateInnerIterations(execute: () => number, targetMilliseconds: number): number {
  let iterations = 1;
  let elapsed = 0;

  while (iterations < 1_048_576 && elapsed < targetMilliseconds) {
    const start = performance.now();
    let sink = 0;
    for (let iteration = 0; iteration < iterations; iteration += 1) sink += execute();
    elapsed = performance.now() - start;
    if (sink === Number.MIN_VALUE) break;
    if (elapsed < targetMilliseconds) iterations *= 2;
  }

  return Math.max(1, iterations);
}

/**
 * Mezcla una colección de forma determinista para reducir sesgo por el orden de
 * ejecución sin sacrificar reproducibilidad del banco.
 */
function deterministicShuffle<T>(items: readonly T[], seed: number): T[] {
  const result = [...items];
  let state = seed >>> 0;

  for (let index = result.length - 1; index > 0; index -= 1) {
    state = (1664525 * state + 1013904223) >>> 0;
    const swapIndex = state % (index + 1);
    [result[index], result[swapIndex]] = [result[swapIndex], result[index]];
  }

  return result;
}

/**
 * Ejecuta un caso para un backend concreto y devuelve un registro auditable.
 */
async function runBenchmarkCase(
  runId: string,
  testCase: BenchmarkCase,
  backend: BenchmarkBackendDefinition,
  configuration: BenchmarkConfiguration,
  uploadedBackends: readonly UploadedBenchmarkAdapterDefinition[],
): Promise<BenchmarkResult> {
  const timestamp = new Date().toISOString();
  const prepared = await prepareBackendCase(backend.id, testCase, uploadedBackends);

  if (!prepared) {
    return {
      id: `${runId}:${backend.id}:${testCase.id}`,
      runId,
      caseId: testCase.id,
      timestamp,
      operation: testCase.operation,
      matrixFormat: testCase.matrixFormat,
      scalarType: testCase.scalarType,
      rows: testCase.rows,
      columns: testCase.columns,
      vectorLength: testCase.vectorLength,
      densityRequested: testCase.densityRequested,
      densityActual: 0,
      nnz: 0,
      ellpackStorageSlots: 0,
      ellpackPaddingRatio: 0,
      warmupIterations: configuration.warmupIterations,
      measuredIterations: configuration.measuredIterations,
      innerIterations: 0,
      statistics: { min: 0, p50: 0, p90: 0, p95: 0, p99: 0, max: 0, mean: 0, standardDeviation: 0, coefficientOfVariation: 0, iqr: 0, confidenceInterval95: 0 },
      estimatedFlops: 0,
      estimatedGflops: 0,
      estimatedMemoryBytes: 0,
      backendId: backend.id,
      backendName: backend.name,
      backendVersion: backend.version,
      environment: backend.environment,
      status: "skipped",
      skipReason: `Backend does not support ${testCase.matrixFormat} for this operation.`,
      checksum: 0,
    };
  }

  let checksum = 0;
  for (let iteration = 0; iteration < configuration.warmupIterations; iteration += 1) checksum += prepared.execute();

  const innerIterations = calibrateInnerIterations(prepared.execute, configuration.targetSampleMilliseconds);
  const samples: number[] = [];

  for (let sampleIndex = 0; sampleIndex < configuration.measuredIterations; sampleIndex += 1) {
    const start = performance.now();
    let sampleSink = 0;
    for (let iteration = 0; iteration < innerIterations; iteration += 1) sampleSink += prepared.execute();
    const elapsed = performance.now() - start;
    checksum += sampleSink;
    samples.push(elapsed / innerIterations);
  }

  const statistics = calculateStatistics(samples);
  const estimatedFlops = estimateFlops(testCase, prepared.nnz);
  const estimatedGflops = statistics.p50 > 0 ? estimatedFlops / (statistics.p50 / 1000) / 1e9 : 0;

  return {
    id: `${runId}:${backend.id}:${testCase.id}`,
    runId,
    caseId: testCase.id,
    timestamp,
    operation: testCase.operation,
    matrixFormat: testCase.matrixFormat,
    scalarType: testCase.scalarType,
    rows: testCase.rows,
    columns: testCase.columns,
    vectorLength: testCase.vectorLength,
    densityRequested: testCase.densityRequested,
    densityActual: prepared.densityActual,
    nnz: prepared.nnz,
    ellpackStorageSlots: prepared.ellpackStorageSlots,
    ellpackPaddingRatio: prepared.ellpackPaddingRatio,
    warmupIterations: configuration.warmupIterations,
    measuredIterations: configuration.measuredIterations,
    innerIterations,
    statistics,
    estimatedFlops,
    estimatedGflops,
    estimatedMemoryBytes: prepared.estimatedMemoryBytes,
    backendId: backend.id,
    backendName: backend.name,
    backendVersion: backend.version,
    environment: backend.environment,
    status: "completed",
    checksum,
  };
}

/**
 * Ejecuta un banco contra el servidor nativo.
 *
 * El cronometro utilizado para la estadistica es el tiempo de kernel que
 * devuelve el servidor, no el tiempo de red. La latencia HTTP queda fuera de
 * la region medida y se registra como coste de transporte en el protocolo.
 */
export async function runServerBenchmarkBank(
  configuration: BenchmarkConfiguration,
  serverOptions: ServerBenchmarkOptions,
  onProgress?: (completed: number, total: number) => void,
): Promise<BenchmarkSnapshot> {
  const runId = `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
  const startedAt = new Date().toISOString();
  const cases = generateBenchmarkCases(configuration);
  const executionCases = deterministicShuffle(cases, 0x5eED1234);
  const total = cases.length * configuration.selectedBackends.length;
  const results: BenchmarkResult[] = [];
  let completed = 0;

  for (const testCase of executionCases) {
    const inputs = createAdapterInputs(testCase);
    const backendDefinitions = NATIVE_BENCHMARK_BACKENDS.filter((backend) => configuration.selectedBackends.includes(backend.id));
    for (const backend of deterministicShuffle(backendDefinitions, testCase.seed)) {
      try {
        if (backend.id === "trace-rust-native-gpu") {
          throw new Error("GPU backend requires the production GPU executor; detection alone is not reported as a GPU measurement.");
        }
        if (backend.id !== "trace-rust-native-cpu") {
          throw new Error(`${backend.name} is not linked into the configured server build.`);
        }

        const matrix = inputs.leftMatrix;
        const rightMatrix = inputs.rightMatrix;
        const request = {
          operation: testCase.operation,
          execution: serverOptions.execution,
          vector: Array.from(inputs.vector),
          other_vector: inputs.otherVector ? Array.from(inputs.otherVector) : undefined,
          scalar: ["vectorScale", "axpy"].includes(testCase.operation) ? 1.25 : 1.0,
          matrix: matrix ? { rows: matrix.rows, columns: matrix.columns, data: Array.from(matrix.data), format: testCase.matrixFormat } : undefined,
          right_matrix: rightMatrix ? { rows: rightMatrix.rows, columns: rightMatrix.columns, data: Array.from(rightMatrix.data), format: "dense" } : undefined,
        };

        const executeRemote = async (): Promise<ServerExecuteResponse> => {
          const response = await fetch(`${serverOptions.serverUrl.replace(/\/$/, "")}/v1/execute`, {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify(request),
          });
          if (!response.ok) {
            const payload = await response.text();
            throw new Error(`Server HTTP ${response.status}: ${payload}`);
          }
          return response.json() as Promise<ServerExecuteResponse>;
        };

        for (let warmup = 0; warmup < configuration.warmupIterations; warmup += 1) await executeRemote();
        const samples: number[] = [];
        let checksum = 0;
        for (let sample = 0; sample < configuration.measuredIterations; sample += 1) {
          const response = await executeRemote();
          samples.push(response.elapsed_ns / 1_000_000);
          checksum += response.checksum;
        }

        const statistics = calculateStatistics(samples);
        const sparseMatrix = testCase.matrixFormat === "csr" ? inputs.leftCsr : testCase.matrixFormat === "csc" ? inputs.leftCsc : testCase.matrixFormat === "ellpack" ? inputs.leftEllpack : undefined;
        const sparseNnz = sparseMatrix ? sparseMatrix.values.length : 0;
        const densityActual = sparseMatrix ? sparseNnz / (testCase.rows * testCase.columns) : testCase.densityRequested;
        const nnz = sparseMatrix ? sparseNnz : testCase.matrixFormat === "dense" && inputs.leftMatrix ? inputs.leftMatrix.data.reduce((count, value) => count + (value !== 0 ? 1 : 0), 0) : 0;
        const ellpackStorageSlots = inputs.leftEllpack && testCase.matrixFormat === "ellpack" ? inputs.leftEllpack.rows * inputs.leftEllpack.maxNonZerosPerRow : 0;
        const ellpackPaddingRatio = ellpackStorageSlots > 0 ? 1 - nnz / ellpackStorageSlots : 0;
        const estimatedFlops = estimateFlops(testCase, nnz);
        const estimatedGflops = statistics.p50 > 0 ? estimatedFlops / (statistics.p50 / 1000) / 1_000_000_000 : 0;

        results.push({
          id: `${runId}:${backend.id}:${testCase.id}`,
          runId,
          caseId: testCase.id,
          timestamp: new Date().toISOString(),
          operation: testCase.operation,
          matrixFormat: testCase.matrixFormat,
          scalarType: testCase.scalarType,
          rows: testCase.rows,
          columns: testCase.columns,
          vectorLength: testCase.vectorLength,
          densityRequested: testCase.densityRequested,
          densityActual,
          nnz,
          ellpackStorageSlots,
          ellpackPaddingRatio,
          warmupIterations: configuration.warmupIterations,
          measuredIterations: configuration.measuredIterations,
          innerIterations: 1,
          statistics,
          estimatedFlops,
          estimatedGflops,
          estimatedMemoryBytes: estimateServerMemoryBytes(inputs),
          backendId: backend.id,
          backendName: backend.name,
          backendVersion: backend.version,
          environment: "server",
          status: "completed",
          checksum,
        });
      } catch (error) {
        results.push({
          id: `${runId}:${backend.id}:${testCase.id}`,
          runId,
          caseId: testCase.id,
          timestamp: new Date().toISOString(),
          operation: testCase.operation,
          matrixFormat: testCase.matrixFormat,
          scalarType: testCase.scalarType,
          rows: testCase.rows,
          columns: testCase.columns,
          vectorLength: testCase.vectorLength,
          densityRequested: testCase.densityRequested,
          densityActual: 0,
          nnz: 0,
          ellpackStorageSlots: 0,
          ellpackPaddingRatio: 0,
          warmupIterations: configuration.warmupIterations,
          measuredIterations: configuration.measuredIterations,
          innerIterations: 1,
          statistics: calculateStatistics([0]),
          estimatedFlops: 0,
          estimatedGflops: 0,
          estimatedMemoryBytes: 0,
          backendId: backend.id,
          backendName: backend.name,
          backendVersion: backend.version,
          environment: "server",
          status: "failed",
          errorMessage: error instanceof Error ? error.message : String(error),
          checksum: 0,
        });
      }
      completed += 1;
      onProgress?.(completed, total);
    }
  }

  const completedAt = new Date().toISOString();
  const run: BenchmarkRun = {
    id: runId,
    startedAt,
    completedAt,
    label: `${configuration.operation} · ${new Date(startedAt).toLocaleString()}`,
    configuration,
    selectedBackends: [...configuration.selectedBackends],
    environment: {
      ...readEnvironmentSnapshot(),
      world: "server",
      platform: "native-server",
    },
    caseCount: cases.length,
    resultCount: results.length,
  };
  return { run, results };
}

function estimateServerMemoryBytes(inputs: BenchmarkAdapterInputs): number {
  let bytes = inputs.vector.byteLength + (inputs.otherVector?.byteLength ?? 0);
  bytes += inputs.leftMatrix?.data.byteLength ?? 0;
  bytes += inputs.rightMatrix?.data.byteLength ?? 0;
  bytes += inputs.leftCsr?.values.byteLength ?? 0;
  bytes += inputs.leftCsr?.columnIndices.byteLength ?? 0;
  bytes += inputs.leftCsr?.rowPointers.byteLength ?? 0;
  bytes += inputs.leftEllpack?.values.byteLength ?? 0;
  bytes += inputs.leftEllpack?.columnIndices.byteLength ?? 0;
  return bytes;
}

/**
 * Ejecuta el banco seleccionado. Un backend no soportado genera un registro
 * skipped para conservar la matriz completa de comparabilidad del run.
 */
export async function runBenchmarkBank(
  configuration: BenchmarkConfiguration,
  onProgress?: (completed: number, total: number) => void,
  uploadedBackends: readonly UploadedBenchmarkAdapterDefinition[] = [],
): Promise<BenchmarkSnapshot> {
  const runId = `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
  const startedAt = new Date().toISOString();
  const cases = generateBenchmarkCases(configuration);
  const uploadedDefinitions = uploadedBackends.map((backend) => ({
    id: backend.id,
    name: backend.name,
    version: backend.version,
    environment: backend.environment,
    description: backend.description,
    operations: backend.operations,
    matrixFormats: backend.matrixFormats,
  } satisfies BenchmarkBackendDefinition));
  const backendRegistry = [...BENCHMARK_BACKENDS, ...uploadedDefinitions];
  const backends = backendRegistry.filter((backend) => configuration.selectedBackends.includes(backend.id));
  const executionCases = deterministicShuffle(cases, 0x5eED1234);
  const total = cases.length * backends.length;
  const results: BenchmarkResult[] = [];
  let completed = 0;

  for (const testCase of executionCases) {
    for (const backend of deterministicShuffle(backends, testCase.seed)) {
      try {
        results.push(await runBenchmarkCase(runId, testCase, backend, configuration, uploadedBackends));
      } catch (error) {
        results.push({
          id: `${runId}:${backend.id}:${testCase.id}`,
          runId,
          caseId: testCase.id,
          timestamp: new Date().toISOString(),
          operation: testCase.operation,
          matrixFormat: testCase.matrixFormat,
          scalarType: testCase.scalarType,
          rows: testCase.rows,
          columns: testCase.columns,
          vectorLength: testCase.vectorLength,
          densityRequested: testCase.densityRequested,
          densityActual: 0,
          nnz: 0,
          ellpackStorageSlots: 0,
          ellpackPaddingRatio: 0,
          warmupIterations: configuration.warmupIterations,
          measuredIterations: configuration.measuredIterations,
          innerIterations: 0,
          statistics: { min: 0, p50: 0, p90: 0, p95: 0, p99: 0, max: 0, mean: 0, standardDeviation: 0, coefficientOfVariation: 0, iqr: 0, confidenceInterval95: 0 },
          estimatedFlops: 0,
          estimatedGflops: 0,
          estimatedMemoryBytes: 0,
          backendId: backend.id,
          backendName: backend.name,
          backendVersion: backend.version,
          environment: backend.environment,
          status: "failed",
          errorMessage: error instanceof Error ? error.message : String(error),
          checksum: 0,
        });
      }

      completed += 1;
      onProgress?.(completed, total);
      if (completed % 2 === 0) await new Promise<void>((resolve) => setTimeout(resolve, 0));
    }
  }

  const completedAt = new Date().toISOString();
  const run: BenchmarkRun = {
    id: runId,
    startedAt,
    completedAt,
    label: `${configuration.operation} · ${new Date(startedAt).toLocaleString()}`,
    configuration,
    selectedBackends: [...configuration.selectedBackends],
    environment: readEnvironmentSnapshot(),
    caseCount: cases.length,
    resultCount: results.length,
  };

  return { run, results };
}
