/**
 * Tipos públicos de la librería independiente de operaciones numéricas.
 *
 * El core no conoce React, Vite, TRACE ni el laboratorio de benchmarking.
 * Esa separación permite reutilizar la librería en navegador, Node.js, WASM
 * y futuros runners nativos sin arrastrar dependencias de la aplicación.
 */

export type ScalarType = "float32" | "float64";

export type MatrixLayout = "row-major";

export type MatrixFormat = "dense" | "csr" | "csc" | "ellpack";

export type OperationName =
  | "vectorAdd"
  | "vectorScale"
  | "dotProduct"
  | "vectorNorm2"
  | "axpy"
  | "crossProduct"
  | "gemv"
  | "matrixVectorMultiply"
  | "triangularSolve"
  | "ger"
  | "matrixMatrixMultiply"
  | "trsm"
  | "syrk"
  | "transpose"
  | "inverse"
  | "determinant"
  | "trace"
  | "matrixNormFrobenius"
  | "matrixNormOne"
  | "matrixNormInf"
  | "matrixNormTwo"
  | "kroneckerProduct"
  | "solveLinearSystem"
  | "solveMultipleRhs"
  | "lu"
  | "cholesky"
  | "ldlt"
  | "bunchKaufman"
  | "qr"
  | "svd"
  | "rankRevealing"
  | "polar"
  | "eigen"
  | "symmetricEigen"
  | "schur"
  | "hessenberg"
  | "jordan"
  | "frobenius"
  | "hermite"
  | "smith"
  | "popov";

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

export interface CscMatrix {
  readonly rows: number;
  readonly columns: number;
  readonly scalarType: ScalarType;
  readonly values: Float32Array | Float64Array;
  readonly rowIndices: Int32Array;
  readonly columnPointers: Int32Array;
}

export interface CsrMatrix {
  readonly rows: number;
  readonly columns: number;
  readonly scalarType: ScalarType;
  readonly values: Float32Array | Float64Array;
  readonly columnIndices: Int32Array;
  readonly rowPointers: Int32Array;
}

export interface EllpackMatrix {
  readonly rows: number;
  readonly columns: number;
  readonly scalarType: ScalarType;
  readonly values: Float32Array | Float64Array;
  readonly columnIndices: Int32Array;
  readonly maxNonZerosPerRow: number;
  readonly nonZeroCount: number;
}

export type Matrix = DenseMatrix | CsrMatrix | CscMatrix | EllpackMatrix;
