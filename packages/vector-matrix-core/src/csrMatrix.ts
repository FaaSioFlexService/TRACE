/**
 * Matriz dispersa CSR (Compressed Sparse Row).
 *
 * CSR almacena únicamente valores no nulos mediante tres buffers contiguos.
 * Es la representación base para SpMV y deja preparado el core para añadir
 * kernels especializados posteriormente.
 */

import type { CsrMatrix, ScalarType, Vector } from "./types";

function createValueArray(
  length: number,
  scalarType: ScalarType,
): Float32Array | Float64Array {
  return scalarType === "float32"
    ? new Float32Array(length)
    : new Float64Array(length);
}

/**
 * Construye una matriz CSR desde una matriz densa.
 *
 * Esta función se utiliza principalmente para generar casos de benchmark y
 * validar la equivalencia entre backends.
 */
export function denseToCsr(
  dense: {
    rows: number;
    columns: number;
    data: Float32Array | Float64Array;
    scalarType: ScalarType;
  },
  zeroTolerance = 0,
): CsrMatrix {
  const values: number[] = [];
  const columnIndices: number[] = [];
  const rowPointers = new Int32Array(dense.rows + 1);

  for (let row = 0; row < dense.rows; row += 1) {
    rowPointers[row] = values.length;
    const offset = row * dense.columns;

    for (let column = 0; column < dense.columns; column += 1) {
      const value = dense.data[offset + column];

      if (Math.abs(value) > zeroTolerance) {
        values.push(value);
        columnIndices.push(column);
      }
    }
  }

  rowPointers[dense.rows] = values.length;

  return {
    rows: dense.rows,
    columns: dense.columns,
    scalarType: dense.scalarType,
    values: createValueArrayFromNumbers(values, dense.scalarType),
    columnIndices: Int32Array.from(columnIndices),
    rowPointers,
  };
}

/**
 * Crea el TypedArray numérico final sin mezclar precisiones.
 */
function createValueArrayFromNumbers(
  values: number[],
  scalarType: ScalarType,
): Float32Array | Float64Array {
  const result = createValueArray(values.length, scalarType);

  for (let index = 0; index < values.length; index += 1) {
    result[index] = values[index];
  }

  return result;
}

/**
 * Calcula CSR * vector.
 */
export function csrMatrixVectorMultiply(
  matrix: CsrMatrix,
  vector: Vector,
): Vector {
  if (matrix.columns !== vector.length) {
    throw new RangeError("Matrix columns must match vector length.");
  }

  const result = createValueArray(matrix.rows, matrix.scalarType);

  for (let row = 0; row < matrix.rows; row += 1) {
    const start = matrix.rowPointers[row];
    const end = matrix.rowPointers[row + 1];
    let accumulator = 0;

    for (let index = start; index < end; index += 1) {
      accumulator += matrix.values[index] * vector.data[matrix.columnIndices[index]];
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
 * Calcula la densidad real de una matriz CSR.
 */
export function csrDensity(matrix: CsrMatrix): number {
  const totalElements = matrix.rows * matrix.columns;
  return totalElements === 0 ? 0 : matrix.values.length / totalElements;
}
