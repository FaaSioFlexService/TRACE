/**
 * Fachada de operaciones matriciales.
 *
 * La selección del backend queda separada de los consumidores. Esto es
 * importante porque el futuro core podrá decidir entre dense, CSR, WASM SIMD,
 * threads o GPU sin modificar la API pública.
 */

import { cscMatrixVectorMultiply } from "./cscMatrix";
import { csrMatrixVectorMultiply } from "./csrMatrix";
import { ellpackMatrixVectorMultiply } from "./ellpackMatrix";
import { denseMatrixMultiply, denseMatrixVectorMultiply } from "./denseMatrix";
import type { Matrix, Vector, DenseMatrix } from "./types";

/**
 * Multiplica una matriz por un vector seleccionando el kernel según su formato.
 */
export function matrixVectorMultiply(
  matrix: Matrix,
  vector: Vector,
): Vector {
  if ("rowPointers" in matrix) {
    return csrMatrixVectorMultiply(matrix, vector);
  }

  if ("columnPointers" in matrix) {
    return cscMatrixVectorMultiply(matrix, vector);
  }

  if ("maxNonZerosPerRow" in matrix) {
    return ellpackMatrixVectorMultiply(matrix, vector);
  }

  return denseMatrixVectorMultiply(matrix, vector);
}

/**
 * Multiplica dos matrices densas.
 *
 * El producto sparse-sparse se deja fuera deliberadamente de esta primera
 * baseline para evitar una API prematura; se añadirá junto con su benchmark.
 */
export function matrixMatrixMultiply(
  left: DenseMatrix,
  right: DenseMatrix,
): DenseMatrix {
  return denseMatrixMultiply(left, right);
}
