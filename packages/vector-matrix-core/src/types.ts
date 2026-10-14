/**
 * Tipos públicos y contratos de la librería independiente de operaciones
 * vectoriales y matriciales.
 *
 * Estos tipos no conocen nada de React, Vite, TRACE ni de la aplicación de
 * benchmarking. Esa separación permite reutilizar el core en otros proyectos.
 */

export type ScalarType = "float32" | "float64";

export type MatrixLayout = "row-major";

export type MatrixFormat = "dense" | "csr";

export type OperationName =
  | "vectorAdd"
  | "vectorScale"
  | "dotProduct"
  | "vectorNorm2"
  | "matrixVectorMultiply"
  | "matrixMatrixMultiply";

export interface Vector {
  readonly length: number;
  readonly data: Float32Array | Float64Array;
  readonly scalarType: ScalarType;
}

export interface DenseMatrix {
  readonly rows: number;
  readonly columns: number;
  readonly layout: MatrixLayout;
  readonly scalarType: ScalarType;
  readonly data: Float32Array | Float64Array;
}

export interface CsrMatrix {
  readonly rows: number;
  readonly columns: number;
  readonly scalarType: ScalarType;
  readonly values: Float32Array | Float64Array;
  readonly columnIndices: Int32Array;
  readonly rowPointers: Int32Array;
}

export type Matrix = DenseMatrix | CsrMatrix;

export interface BenchmarkSample {
  readonly id: string;
  readonly timestamp: string;
  readonly operation: OperationName;
  readonly matrixFormat: MatrixFormat | "none";
  readonly scalarType: ScalarType;
  readonly rows: number;
  readonly columns: number;
  readonly vectorLength: number;
  readonly density: number;
  readonly warmupIterations: number;
  readonly measuredIterations: number;
  readonly elapsedMilliseconds: number;
  readonly medianMilliseconds: number;
  readonly meanMilliseconds: number;
  readonly minMilliseconds: number;
  readonly maxMilliseconds: number;
  readonly standardDeviationMilliseconds: number;
  readonly estimatedFlops: number;
  readonly estimatedGflops: number;
  readonly estimatedMemoryBytes: number;
  readonly backend: string;
}

export interface BenchmarkConfiguration {
  readonly operation: OperationName;
  readonly matrixFormats: MatrixFormat[];
  readonly scalarTypes: ScalarType[];
  readonly sizes: number[];
  readonly densities: number[];
  readonly warmupIterations: number;
  readonly measuredIterations: number;
  readonly casesPerConfiguration: number;
  readonly backend: string;
}
