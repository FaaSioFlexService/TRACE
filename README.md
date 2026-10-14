# TRACE HPC Vector/Matrix Benchmark

Prototype de la infraestructura de benchmarking para la futura librería independiente de álgebra lineal de alto rendimiento.

## Objetivo

La aplicación permite:

- Crear bancos de pruebas automáticamente.
- Generar vectores y matrices deterministas o aleatorios.
- Comparar representaciones `dense-row-major` y `CSR`.
- Ejecutar operaciones vectoriales y matriciales.
- Medir tiempo de cómputo con `performance.now()`.
- Registrar cada ejecución en un historial.
- Calcular estadísticas de rendimiento.
- Exportar resultados en JSON y CSV.
- Generar un informe técnico reproducible.

## Arquitectura

```text
trace-hpc-benchmark/
├── packages/
│   └── vector-matrix-core/
│       └── src/
│           ├── types.ts
│           ├── random.ts
│           ├── vector.ts
│           ├── denseMatrix.ts
│           ├── csrMatrix.ts
│           ├── operations.ts
│           └── index.ts
└── apps/
    └── benchmark-web/
        └── src/
            ├── App.tsx
            ├── benchmark.ts
            ├── main.tsx
            └── styles.css
```

La aplicación React/Vite depende de la librería mediante workspace. La librería no depende de React, Vite ni de la UI.

## Decisiones iniciales

Esta primera iteración es deliberadamente CPU/TypeScript para crear una línea base fiable. No se debe introducir WASM, SIMD o workers antes de disponer de un benchmark de referencia.

La siguiente fase puede añadir backends intercambiables:

```text
Core API
   │
   ├── JS/TS scalar baseline
   ├── WASM SIMD
   ├── WASM threads
   └── WebGPU
```

Así podremos medir cada optimización contra una referencia estable.

## Ejecutar

```bash
npm install
npm run dev
```

Después abrir la URL mostrada por Vite.

## Filosofía del benchmark

No se deben comparar únicamente "operaciones por segundo". El banco guarda:

- operación
- representación
- precisión
- dimensiones
- número de elementos
- densidad
- iteraciones
- warmup
- min/median/mean/max
- desviación estándar
- GFLOP/s estimados
- memoria estimada
- backend

Esto permitirá utilizar posteriormente los mismos datos para comparar la implementación propia con WASM, WebGPU y referencias externas.
