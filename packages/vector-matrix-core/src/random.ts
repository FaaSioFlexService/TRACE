/**
 * Generadores deterministas y aleatorios.
 *
 * El generador determinista permite repetir exactamente un banco de pruebas
 * y comparar versiones de la librería sin introducir ruido por los datos.
 */

export class SeededRandom {
  private state: number;

  /**
   * Inicializa un generador pseudoaleatorio reproducible.
   */
  public constructor(seed = 0x9e3779b9) {
    this.state = seed >>> 0;
  }

  /**
   * Devuelve un número pseudoaleatorio en [0, 1).
   */
  public next(): number {
    this.state = (1664525 * this.state + 1013904223) >>> 0;
    return this.state / 0x100000000;
  }

  /**
   * Devuelve un número en el intervalo [min, max).
   */
  public range(min: number, max: number): number {
    return min + (max - min) * this.next();
  }

  /**
   * Devuelve un número con distribución aproximadamente normal.
   */
  public normal(mean = 0, standardDeviation = 1): number {
    const u1 = Math.max(this.next(), Number.MIN_VALUE);
    const u2 = this.next();
    const magnitude = Math.sqrt(-2 * Math.log(u1));
    return mean + standardDeviation * magnitude * Math.cos(2 * Math.PI * u2);
  }
}

/**
 * Crea un TypedArray del tipo numérico solicitado y lo rellena.
 */
export function createRandomData(
  length: number,
  scalarType: "float32" | "float64",
  seed: number,
  minimum = -1,
  maximum = 1,
): Float32Array | Float64Array {
  const random = new SeededRandom(seed);
  const data = scalarType === "float32"
    ? new Float32Array(length)
    : new Float64Array(length);

  for (let index = 0; index < length; index += 1) {
    data[index] = random.range(minimum, maximum);
  }

  return data;
}
