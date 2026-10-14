/**
 * Motor de generación y ejecución de bancos de pruebas.
 *
 * Esta capa pertenece a la aplicación de benchmarking, no a la librería core.
 * Su responsabilidad es crear casos reproducibles, medirlos y convertirlos
 * en registros comparables.
 */

import {
  createDenseMatrix,
  createRandomData,
  createVector,
  denseToCsr,
  matrixMatrixMultiply,
  matrixVectorMultiply,
  vectorAdd,
  vectorNorm2,
  vectorScale,
  dotProduct,
  type BenchmarkConfiguration,
  type BenchmarkSample,
  type DenseMatrix,
  type MatrixFormat,
  type OperationName,
  type ScalarType,
} from "@trace/vector-matrix-core";

export interface BenchmarkCase {
  readonly id: string;
  readonly operation: OperationName;
  readonly matrixFormat: MatrixFormat | "none";
  readonly scalarType: ScalarType;
  readonly size: number;
  readonly rows: number;
  readonly columns: number;
  readonly vectorLength: number;
  readonly density: number;
  readonly seed: number;
}

/**
 * Crea una matriz densa aleatoria con densidad controlada.
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

  if (density < 1) {
    const keepThreshold = density;
    let state = (seed ^ 0xa5a5a5a5) >>> 0;

    for (let index = 0; index < matrix.data.length; index += 1) {
      state = (1664525 * state + 1013904223) >>> 0;
      const probability = state / 0x100000000;

      if (probability > keepThreshold) {
        matrix.data[index] = 0;
      }
    }
  }

  return matrix;
}

/**
 * Genera el banco cartesiano solicitado por el usuario.
 */
export function generateBenchmarkCases(
  configuration: BenchmarkConfiguration,
): BenchmarkCase[] {
  const cases: BenchmarkCase[] = [];
  let caseNumber = 0;

  for (const scalarType of configuration.scalarTypes) {
    for (const size of configuration.sizes) {
      for (const density of configuration.densities) {
        const formats =
          configuration.operation.startsWith("matrix")
            ? configuration.matrixFormats
            : ["none" as const];

        for (const matrixFormat of formats) {
          for (let repetition = 0; repetition < configuration.casesPerConfiguration; repetition += 1) {
            const rows = size;
            const columns = size;
            const vectorLength = size;

            cases.push({
              id: `case-${caseNumber++}`,
              operation: configuration.operation,
              matrixFormat,
              scalarType,
              size,
              rows,
              columns,
              vectorLength,
              density,
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
 * Ejecuta una operación concreta y devuelve un valor observable.
 *
 * El resultado se consume para impedir que el benchmark mida una ruta que el
 * motor pueda optimizar de forma inesperada en futuras implementaciones.
 */
function executeCase(testCase: BenchmarkCase): number {
  const vectorData = createRandomData(
    testCase.vectorLength,
    testCase.scalarType,
    testCase.seed + 1,
  );
  const vector = createVector(vectorData, testCase.scalarType);

  if (testCase.operation === "vectorAdd") {
    const other = createVector(
      createRandomData(testCase.vectorLength, testCase.scalarType, testCase.seed + 2),
      testCase.scalarType,
    );
    return vectorAdd(vector, other).data[0];
  }

  if (testCase.operation === "vectorScale") {
    return vectorScale(vector, 1.25).data[0];
  }

  if (testCase.operation === "dotProduct") {
    return dotProduct(vector, vector);
  }

  if (testCase.operation === "vectorNorm2") {
    return vectorNorm2(vector);
  }

  const left = createRandomDenseMatrix(
    testCase.rows,
    testCase.columns,
    testCase.scalarType,
    testCase.density,
    testCase.seed + 3,
  );

  if (testCase.operation === "matrixVectorMultiply") {
    const matrix =
      testCase.matrixFormat === "csr" ? denseToCsr(left) : left;
    return matrixVectorMultiply(matrix, vector).data[0];
  }

  const right = createRandomDenseMatrix(
    testCase.columns,
    testCase.columns,
    testCase.scalarType,
    testCase.density,
    testCase.seed + 4,
  );

  return matrixMatrixMultiply(left, right).data[0];
}

/**
 * Estima FLOPs para mostrar una métrica comparable entre ejecuciones.
 */
function estimateFlops(testCase: BenchmarkCase): number {
  if (testCase.operation === "vectorAdd") return testCase.size;
  if (testCase.operation === "vectorScale") return testCase.size;
  if (testCase.operation === "dotProduct") return 2 * testCase.size;
  if (testCase.operation === "vectorNorm2") return 2 * testCase.size;
  if (testCase.operation === "matrixVectorMultiply") {
    return 2 * testCase.rows * testCase.columns * testCase.density;
  }
  return 2 * testCase.rows * testCase.columns * testCase.columns * testCase.density;
}

/**
 * Estima memoria principal del caso para facilitar el análisis de eficiencia.
 */
function estimateMemoryBytes(testCase: BenchmarkCase): number {
  const scalarBytes = testCase.scalarType === "float32" ? 4 : 8;

  if (testCase.operation.startsWith("vector")) {
    return testCase.vectorLength * scalarBytes;
  }

  if (testCase.matrixFormat === "csr") {
    const nonZeros = Math.ceil(testCase.rows * testCase.columns * testCase.density);
    return nonZeros * scalarBytes + nonZeros * 4 + (testCase.rows + 1) * 4;
  }

  return testCase.rows * testCase.columns * scalarBytes;
}

/**
 * Calcula estadísticas robustas sobre una serie de tiempos.
 */
function calculateStatistics(values: number[]) {
  const sorted = [...values].sort((left, right) => left - right);
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance =
    values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;

  return {
    min: sorted[0],
    median: sorted[Math.floor(sorted.length / 2)],
    mean,
    max: sorted[sorted.length - 1],
    standardDeviation: Math.sqrt(variance),
  };
}

/**
 * Ejecuta y mide un caso con warmup separado de las muestras.
 */
export function runBenchmarkCase(
  testCase: BenchmarkCase,
  warmupIterations: number,
  measuredIterations: number,
  backend = "typescript-baseline",
): BenchmarkSample {
  for (let iteration = 0; iteration < warmupIterations; iteration += 1) {
    executeCase(testCase);
  }

  const samples: number[] = [];

  for (let iteration = 0; iteration < measuredIterations; iteration += 1) {
    const start = performance.now();
    executeCase(testCase);
    samples.push(performance.now() - start);
  }

  const statistics = calculateStatistics(samples);
  const flops = estimateFlops(testCase);
  const medianSeconds = statistics.median / 1000;

  return {
    id: testCase.id,
    timestamp: new Date().toISOString(),
    operation: testCase.operation,
    matrixFormat: testCase.matrixFormat,
    scalarType: testCase.scalarType,
    rows: testCase.rows,
    columns: testCase.columns,
    vectorLength: testCase.vectorLength,
    density: testCase.density,
    warmupIterations,
    measuredIterations,
    elapsedMilliseconds: statistics.mean,
    medianMilliseconds: statistics.median,
    meanMilliseconds: statistics.mean,
    minMilliseconds: statistics.min,
    maxMilliseconds: statistics.max,
    standardDeviationMilliseconds: statistics.standardDeviation,
    estimatedFlops: flops,
    estimatedGflops: medianSeconds > 0 ? flops / medianSeconds / 1e9 : 0,
    estimatedMemoryBytes: estimateMemoryBytes(testCase),
    backend,
  };
}

/**
 * Ejecuta un banco completo y permite actualizar progreso.
 */
export async function runBenchmarkBank(
  configuration: BenchmarkConfiguration,
  onProgress?: (completed: number, total: number) => void,
): Promise<BenchmarkSample[]> {
  const cases = generateBenchmarkCases(configuration);
  const results: BenchmarkSample[] = [];

  for (let index = 0; index < cases.length; index += 1) {
    results.push(
      runBenchmarkCase(
        cases[index],
        configuration.warmupIterations,
        configuration.measuredIterations,
        configuration.backend,
      ),
    );

    onProgress?.(index + 1, cases.length);

    // Cede el control al navegador para que la UI permanezca actualizable.
    if (index % 2 === 1) {
      await new Promise<void>((resolve) => setTimeout(resolve, 0));
    }
  }

  return results;
}
