/**
 * Matriz dispersa ELLPACK (ELLPACK/ITPACK).
 *
 * ELLPACK almacena un número fijo de posiciones por fila, determinado por
 * la fila con más elementos no nulos. Esta regularidad simplifica el acceso
 * durante SpMV y deja una estructura especialmente adecuada para futuras
 * optimizaciones SIMD/GPU. Su coste es el padding adicional en matrices con
 * una distribución muy irregular de NNZ por fila.
 */

import type { EllpackMatrix, ScalarType, Vector } from "./types";

/**
 * Crea el TypedArray de valores con la precisión solicitada.
 */
function createValueArray(
  length: number,
  scalarType: ScalarType,
): Float32Array | Float64Array {
  return scalarType === "float32"
    ? new Float32Array(length)
    : new Float64Array(length);
}

/**
 * Construye una matriz ELLPACK a partir de una matriz densa.
 *
 * El almacenamiento es row-major lógico: para cada fila se reservan
 * `maxNonZerosPerRow` posiciones. Las posiciones de padding usan columna -1
 * y valor 0, por lo que el kernel puede recorrer un ancho fijo sin acceder
 * fuera del vector de entrada.
 */
export function denseToEllpack(
  dense: {
    rows: number;
    columns: number;
    data: Float32Array | Float64Array;
    scalarType: ScalarType;
  },
  zeroTolerance = 0,
): EllpackMatrix {
  let maxNonZerosPerRow = 0;
  let nonZeroCount = 0;

  // Primera pasada: medir la estructura real para dimensionar el padding.
  for (let row = 0; row < dense.rows; row += 1) {
    const offset = row * dense.columns;
    let rowCount = 0;

    for (let column = 0; column < dense.columns; column += 1) {
      if (Math.abs(dense.data[offset + column]) > zeroTolerance) {
        rowCount += 1;
      }
    }

    nonZeroCount += rowCount;
    maxNonZerosPerRow = Math.max(maxNonZerosPerRow, rowCount);
  }

  const storageLength = dense.rows * maxNonZerosPerRow;
  const values = createValueArray(storageLength, dense.scalarType);
  const columnIndices = new Int32Array(storageLength);

  // -1 identifica las posiciones de padding y evita confundirlas con la
  // columna 0, que es un índice válido.
  columnIndices.fill(-1);

  // Segunda pasada: empaquetar cada fila en un ancho fijo.
  for (let row = 0; row < dense.rows; row += 1) {
    const denseOffset = row * dense.columns;
    const ellpackOffset = row * maxNonZerosPerRow;
    let slot = 0;

    for (let column = 0; column < dense.columns; column += 1) {
      const value = dense.data[denseOffset + column];

      if (Math.abs(value) <= zeroTolerance) {
        continue;
      }

      values[ellpackOffset + slot] = value;
      columnIndices[ellpackOffset + slot] = column;
      slot += 1;
    }
  }

  return {
    rows: dense.rows,
    columns: dense.columns,
    scalarType: dense.scalarType,
    values,
    columnIndices,
    maxNonZerosPerRow,
    nonZeroCount,
  };
}

/**
 * Calcula ELLPACK * vector.
 *
 * El kernel recorre siempre el mismo número de posiciones por fila. Las
 * posiciones con columna -1 son padding y se omiten. Esta regularidad es la
 * principal diferencia frente a CSR y será útil para comparar acceso lineal,
 * SIMD y futuras implementaciones sobre GPU.
 */
export function ellpackMatrixVectorMultiply(
  matrix: EllpackMatrix,
  vector: Vector,
): Vector {
  if (matrix.columns !== vector.length) {
    throw new RangeError("Matrix columns must match vector length.");
  }

  const result = createValueArray(matrix.rows, matrix.scalarType);

  for (let row = 0; row < matrix.rows; row += 1) {
    const offset = row * matrix.maxNonZerosPerRow;
    let accumulator = 0;

    for (let slot = 0; slot < matrix.maxNonZerosPerRow; slot += 1) {
      const column = matrix.columnIndices[offset + slot];

      if (column < 0) {
        continue;
      }

      accumulator += matrix.values[offset + slot] * vector.data[column];
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
 * Calcula la densidad real de una matriz ELLPACK.
 */
export function ellpackDensity(matrix: EllpackMatrix): number {
  const totalElements = matrix.rows * matrix.columns;
  return totalElements === 0 ? 0 : matrix.nonZeroCount / totalElements;
}
