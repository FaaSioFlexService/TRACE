/**
 * Operaciones vectoriales de referencia.
 *
 * Estas implementaciones constituyen el backend baseline. Posteriormente se
 * podrán mantener exactamente estos contratos y sustituir internamente el
 * kernel por WASM SIMD, WebGPU u otro backend.
 */

import type { ScalarType, Vector } from "./types";

function createTypedArray(
  length: number,
  scalarType: ScalarType,
): Float32Array | Float64Array {
  return scalarType === "float32"
    ? new Float32Array(length)
    : new Float64Array(length);
}

/**
 * Crea un vector a partir de datos tipados.
 */
export function createVector(
  data: Float32Array | Float64Array,
  scalarType: ScalarType,
): Vector {
  return {
    length: data.length,
    data,
    scalarType,
  };
}

/**
 * Suma dos vectores elemento a elemento.
 */
export function vectorAdd(left: Vector, right: Vector): Vector {
  if (left.length !== right.length) {
    throw new RangeError("Vector lengths must match.");
  }

  const result = createTypedArray(left.length, left.scalarType);

  for (let index = 0; index < left.length; index += 1) {
    result[index] = left.data[index] + right.data[index];
  }

  return createVector(result, left.scalarType);
}

/**
 * Multiplica un vector por un escalar.
 */
export function vectorScale(vector: Vector, scalar: number): Vector {
  const result = createTypedArray(vector.length, vector.scalarType);

  for (let index = 0; index < vector.length; index += 1) {
    result[index] = vector.data[index] * scalar;
  }

  return createVector(result, vector.scalarType);
}

/**
 * Calcula el producto escalar.
 *
 * Se mantiene un acumulador escalar para disponer de una baseline clara antes
 * de introducir técnicas como pairwise reduction o SIMD.
 */
export function dotProduct(left: Vector, right: Vector): number {
  if (left.length !== right.length) {
    throw new RangeError("Vector lengths must match.");
  }

  let result = 0;

  for (let index = 0; index < left.length; index += 1) {
    result += left.data[index] * right.data[index];
  }

  return result;
}

/**
 * Calcula la norma euclídea.
 */
export function vectorNorm2(vector: Vector): number {
  return Math.sqrt(dotProduct(vector, vector));
}
