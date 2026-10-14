/**
 * Matriz densa row-major.
 *
 * Row-major se adopta como layout canónico del baseline porque permite recorrer
 * cada fila secuencialmente en matvec y proporciona una referencia estable para
 * futuras optimizaciones de caché, blocking y SIMD.
 */

import type { DenseMatrix, ScalarType, Vector } from "./types";

function createTypedArray(
  length: number,
  scalarType: ScalarType,
): Float32Array | Float64Array {
  return scalarType === "float32"
    ? new Float32Array(length)
    : new Float64Array(length);
}

/**
 * Crea una matriz densa row-major.
 */
export function createDenseMatrix(
  rows: number,
  columns: number,
  scalarType: ScalarType,
  data?: Float32Array | Float64Array,
): DenseMatrix {
  const matrixData = data ?? createTypedArray(rows * columns, scalarType);

  if (matrixData.length !== rows * columns) {
    throw new RangeError("Matrix data length does not match dimensions.");
  }

  return {
    rows,
    columns,
    layout: "row-major",
    scalarType,
    data: matrixData,
  };
}

/**
 * Lee un elemento de la matriz.
 */
export function getDenseElement(
  matrix: DenseMatrix,
  row: number,
  column: number,
): number {
  return matrix.data[row * matrix.columns + column];
}

/**
 * Escribe un elemento de la matriz.
 */
export function setDenseElement(
  matrix: DenseMatrix,
  row: number,
  column: number,
  value: number,
): void {
  matrix.data[row * matrix.columns + column] = value;
}

/**
 * Multiplica una matriz densa por un vector.
 */
export function denseMatrixVectorMultiply(
  matrix: DenseMatrix,
  vector: Vector,
): Vector {
  if (matrix.columns !== vector.length) {
    throw new RangeError("Matrix columns must match vector length.");
  }

  const result = createTypedArray(matrix.rows, matrix.scalarType);

  for (let row = 0; row < matrix.rows; row += 1) {
    const offset = row * matrix.columns;
    let accumulator = 0;

    for (let column = 0; column < matrix.columns; column += 1) {
      accumulator += matrix.data[offset + column] * vector.data[column];
    }

    result[row] = accumulator;
  }

  return {
    length: matrix.rows,
    data: result,
    scalarType: matrix.scalarType,
  };
}

/**
 * Multiplica dos matrices densas row-major.
 *
 * El bucle se organiza para reutilizar el elemento de la fila izquierda y
 * recorrer secuencialmente la fila de la matriz derecha.
 */
export function denseMatrixMultiply(
  left: DenseMatrix,
  right: DenseMatrix,
): DenseMatrix {
  if (left.columns !== right.rows) {
    throw new RangeError("Matrix dimensions are incompatible.");
  }

  const result = createDenseMatrix(
    left.rows,
    right.columns,
    left.scalarType,
  );

  for (let row = 0; row < left.rows; row += 1) {
    for (let shared = 0; shared < left.columns; shared += 1) {
      const leftValue = left.data[row * left.columns + shared];
      const rightOffset = shared * right.columns;
      const resultOffset = row * right.columns;

      for (let column = 0; column < right.columns; column += 1) {
        result.data[resultOffset + column] +=
          leftValue * right.data[rightOffset + column];
      }
    }
  }

  return result;
}
