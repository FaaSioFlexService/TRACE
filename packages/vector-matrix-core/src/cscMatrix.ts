/**
 * Matriz dispersa CSC (Compressed Sparse Column).
 *
 * CSC complementa CSR: mantiene columnas contiguas y es especialmente útil
 * para operaciones orientadas a columnas y productos con la matriz transpuesta.
 */
import type { CscMatrix, ScalarType, Vector } from "./types";

function createValueArray(length: number, scalarType: ScalarType): Float32Array | Float64Array {
  return scalarType === "float32" ? new Float32Array(length) : new Float64Array(length);
}

/**
 * Construye CSC desde una matriz densa sin alterar el buffer de entrada.
 */
export function denseToCsc(
  dense: { rows: number; columns: number; data: Float32Array | Float64Array; scalarType: ScalarType },
  zeroTolerance = 0,
): CscMatrix {
  const values: number[] = [];
  const rowIndices: number[] = [];
  const columnPointers = new Int32Array(dense.columns + 1);

  for (let column = 0; column < dense.columns; column += 1) {
    columnPointers[column] = values.length;
    for (let row = 0; row < dense.rows; row += 1) {
      const value = dense.data[row * dense.columns + column];
      if (Math.abs(value) > zeroTolerance) {
        values.push(value);
        rowIndices.push(row);
      }
    }
  }
  columnPointers[dense.columns] = values.length;

  const typedValues = createValueArray(values.length, dense.scalarType);
  typedValues.set(values);
  return {
    rows: dense.rows,
    columns: dense.columns,
    scalarType: dense.scalarType,
    values: typedValues,
    rowIndices: Int32Array.from(rowIndices),
    columnPointers,
  };
}

/**
 * Calcula CSC * vector recorriendo columnas contiguas.
 */
export function cscMatrixVectorMultiply(matrix: CscMatrix, vector: Vector): Vector {
  if (matrix.columns !== vector.length) throw new RangeError("Matrix columns must match vector length.");
  const result = createValueArray(matrix.rows, matrix.scalarType);
  for (let column = 0; column < matrix.columns; column += 1) {
    const value = vector.data[column];
    for (let index = matrix.columnPointers[column]; index < matrix.columnPointers[column + 1]; index += 1) {
      result[matrix.rowIndices[index]] += matrix.values[index] * value;
    }
  }
  return { length: matrix.rows, data: result, scalarType: matrix.scalarType };
}

/**
 * Calcula la densidad real de una matriz CSC.
 */
export function cscDensity(matrix: CscMatrix): number {
  const totalElements = matrix.rows * matrix.columns;
  return totalElements === 0 ? 0 : matrix.values.length / totalElements;
}
