/**
 * Declaraciones mínimas para los paquetes CommonJS del ecosistema scijs.
 *
 * El adaptador usa únicamente las funciones documentadas por cada paquete.
 */
declare module "ndarray" {
  interface NdArray<T = unknown> {
    readonly data: T;
    readonly shape: number[];
    readonly stride: number[];
    readonly offset: number;
    readonly size: number;
    get(...indices: number[]): number;
    set(...indices: number[]): void;
  }

  function ndarray<T>(data: T, shape?: number[], stride?: number[], offset?: number): NdArray<T>;
  export default ndarray;
}

declare module "ndarray-ops" {
  import type { NdArray } from "ndarray";
  const ndarrayOps: { norm2(array: NdArray): number };
  export default ndarrayOps;
}

declare module "ndarray-blas-gemv" {
  import type { NdArray } from "ndarray";
  function gemv(alpha: number, matrix: NdArray, vector: NdArray, beta: number, output: NdArray): void;
  export default gemv;
}
