# TRACE HPC Benchmark Lab

Benchmark laboratory for the independent vector/matrix numerical core used by the TFM project.

## v0.3.2 UI refinement

The benchmark workbench now uses a fixed no-scroll application shell, a compact always-visible test-bank sidebar, external pagination for large result sets, campaign/slice selectors for charts, explicit TRACE ownership labels, stable per-backend colors, and persistent light/dark themes.

## v0.3.0 architecture

The application is now a fixed-size web workspace instead of a long scrolling document:

```text
┌─────────────────────────────────────────────────────────────────────┐
│ TRACE HPC BENCHMARK LAB                                             │
├──────────────────────┬──────────────────────────────────────────────┤
│ Persistent config    │ My Library | Comparison | Charts | History  │
│                      │ Methodology | Environment                    │
│ operation            │                                              │
│ formats              │ Active tab with internal scrolling           │
│ precision             │                                              │
│ sizes                 │                                              │
│ densities             │                                              │
│ backend checkboxes   │                                              │
│ warmup / samples     │                                              │
│ run                  │                                              │
└──────────────────────┴──────────────────────────────────────────────┘
```

The browser page itself does not scroll. The sidebar and workspace have independent internal scroll regions.

## Benchmark campaign model

A benchmark is stored as an immutable campaign:

```text
BenchmarkRun
├── configuration
├── selectedBackends
├── environment
├── caseCount
└── results[]
```

Each result is keyed by:

```text
runId + backendId + caseId
```

This prevents collisions between repeated campaigns and makes comparison reproducible.

### Why campaign-scoped comparison?

If campaign A executes TRACE + ml-matrix and campaign B executes TRACE + JavaScript baseline, the UI does **not** allow campaign A to display JavaScript results as if they belonged to it. The Comparison tab creates a sub-tab for every stored campaign and exposes only the backends that were actually executed in that campaign.

This is deliberate: a benchmark cannot visualize or compare a backend that has never produced a measurement for the selected campaign.

## Current browser backends

| Backend | Environment | Dense | CSR | ELLPACK |
|---|---|---:|---:|---:|
| TRACE Matrix Core | Browser/TypeScript | yes | yes | yes |
| JavaScript baseline | Browser | yes | no | no |
| ml-matrix | Browser | yes | no | no |
| ndarray + BLAS GEMV | Browser | yes (GEMV) | no | no |

Unsupported combinations are stored as `skipped`, never fabricated as zero or as a measurement from another implementation.

The registry is designed so additional adapters can be added without modifying the UI or benchmark engine.

## Sparse formats

The core currently contains:

- CSR for irregular sparse rows and efficient row-oriented SpMV.
- CSC for column-oriented traversal and transpose-friendly workloads.
- ELLPACK for regular row structure and future SIMD/GPU-oriented experiments.

For sparse results the benchmark records:

- requested density;
- actual density;
- NNZ;
- ELLPACK storage slots;
- ELLPACK padding ratio;
- estimated working-set size.

SpMV FLOPs are based on actual NNZ, not only requested density.

## Timing methodology

The measured region excludes:

- random data generation;
- object construction;
- Dense → CSR conversion;
- Dense → ELLPACK conversion;
- adapter setup.

Each sample executes an automatically calibrated number of inner kernel iterations so very small operations do not collapse into `0.00 ms` due to timer granularity.

The stored statistics are:

- minimum;
- P50;
- P90;
- P95;
- P99;
- maximum;
- mean;
- standard deviation;
- coefficient of variation;
- IQR;
- approximate 95% confidence interval.

## Comparison views

### My Library

Contains only TRACE Matrix Core results for the active campaign.

### Comparison

Contains campaign sub-tabs and competitor checkboxes. Selecting/unselecting a competitor changes only visualization; it does not modify stored results.

The comparison table shows:

- same case;
- same operation;
- same size;
- same format;
- TRACE P50;
- competitor P50;
- competitor/TRACE relative runtime.

### Charts

The current SVG chart layer contains:

- P50 runtime vs problem size;
- GFLOP/s vs problem size;
- relative performance vs TRACE.

The chart layer is deliberately independent from the benchmark engine so additional plots can be added later without touching the measurement code.

## Future native/HPC plane

Native libraries must not be mixed directly with browser measurements. The architecture is intended to add a second benchmark plane later:

```text
Browser
├── TRACE TypeScript
├── JavaScript baseline
├── TRACE WASM
├── nalgebra WASM
├── faer WASM
└── other browser-compatible adapters

Native
├── TRACE Rust core
├── OpenBLAS
├── BLIS
├── oneMKL
├── Eigen
└── other native references
```

The common benchmark case model allows the two planes to export comparable records while preserving their execution environments.

## Installation

```bash
npm install
npm run build
npm run dev
```

The `ml-matrix` and `ndarray`/BLAS GEMV adapters are part of the web application's dependencies.

## Project structure

```text
packages/vector-matrix-core/
└── src/
    ├── denseMatrix.ts
    ├── csrMatrix.ts
    ├── ellpackMatrix.ts
    ├── operations.ts
    ├── random.ts
    ├── types.ts
    ├── vector.ts
    └── index.ts

apps/benchmark-web/
└── src/
    ├── App.tsx
    ├── benchmark.ts
    ├── history.ts
    ├── charts/
    │   └── BenchmarkCharts.tsx
    ├── components/
    │   ├── BackendSelector.tsx
    │   ├── DataTable.tsx
    │   ├── FilterBar.tsx
    │   ├── MetricCard.tsx
    │   ├── Pagination.tsx
    │   ├── RunTabs.tsx
    │   ├── Tabs.tsx
    │   └── index.ts
    └── styles.css
```

## v0.3.3 — Professional benchmark reporting and external adapters

### Historical campaign selector

The first **Benchmark** tab now contains a campaign selector. It is initialized to the most recent stored campaign and can switch between every historical battery without opening the History tab.

### Professional metrics

The Charts section is organized into four views so the fixed viewport remains readable:

- **Overview:** P50 runtime, estimated GFLOP/s, relative speedup and P50 with confidence interval.
- **Stability:** coefficient of variation, P95/P50 tail ratio, confidence intervals and relative runtime.
- **Memory:** estimated working set, runtime, GFLOP/s and relative speedup.
- **Statistics:** P50, P95/P50, CV and confidence intervals.

The purpose is not to produce a single synthetic score. Each metric answers a different engineering question: throughput, latency, memory footprint, variability and tail behavior.

This follows the methodology used by professional benchmark frameworks: dynamic timing, warmup, repeated measurements, metadata/context, multiple output statistics and explicit result comparison. Google Benchmark documents minimum-time execution, warmup, repetitions, context metadata, JSON/CSV output, random interleaving and aggregate statistics. Criterion.rs likewise emphasizes statistics-driven regression detection and charts. 

### External library adapter contract

The laboratory intentionally does **not** attempt to infer an arbitrary library's API and generate opaque calls automatically. That would make the benchmark non-reproducible and unsafe: two libraries can expose similar operations with incompatible storage, mutation, initialization or lifetime semantics.

Instead, v0.3.3 introduces a stable adapter contract. The user uploads a JSON bundle containing:

```text
metadata
└── id / name / version / environment
    operations[]
    matrixFormats[]

module
└── createAdapter()
    └── prepare({ testCase, inputs })
        └── execute()
```

The adapter receives canonical deterministic inputs generated from the benchmark case. Its `prepare()` method must perform all setup outside the timed region and return an `execute()` kernel plus metadata such as actual density, NNZ and estimated memory.

Uploaded adapter code is executed only inside the Benchmark Worker, never on the React UI thread. The worker boundary prevents a bad adapter from freezing the interface, but it is **not** a security boundary for hostile code. Only trusted adapters should be uploaded. WebAssembly provides a stronger sandboxing model and will be the preferred future contract for third-party WASM libraries.

The UI provides a **Template** button that generates a complete adapter JSON example.

### Browser vs Native/HPC

The browser plane remains separate from the future native plane. Native OpenBLAS, BLIS, oneMKL, Eigen and Rust implementations should not be mixed directly with browser measurements because the runtime, memory model and execution environment differ.

```text
Benchmark Engine
├── Browser
│   ├── TRACE Matrix Core
│   ├── JavaScript baseline
│   ├── ml-matrix
│   ├── ndarray + BLAS GEMV
│   └── uploaded browser adapters
│
└── Native/HPC (future)
    ├── TRACE Rust
    ├── OpenBLAS
    ├── BLIS
    ├── oneMKL
    ├── Eigen
    └── faer / nalgebra
```

## v0.3.4 — metric explainability and ES/EN UI

The comparison UI now explains the meaning and origin of numerical values directly from the interface. **P50 means the 50th percentile (median) of the measured samples**; it is not an operation. The comparison ratio is explicitly named **Relative time (Library / TRACE)** and is calculated as:

`relative = libraryP50 / traceP50`

Interpretation:
- `1.00×`: same P50 runtime.
- `< 1.00×`: the compared library is faster.
- `> 1.00×`: the compared library is slower.

P50, P95, relative time, GFLOP/s, CV, confidence interval, working set and chart values expose contextual explanations on hover/focus. The application also has a persistent **ES / EN** selector in the top-right corner. Technical backend identifiers remain unchanged for auditability.

## v0.4.0 - Browser and native HPC worlds

The project now separates two execution worlds:

- **Browser / Offline**: the existing JavaScript/WASM benchmark environment.
- **Server / Native**: a Rust-native numerical core and native benchmark server.

The Rust core includes runtime CPU detection, SIMD dispatch, automatic CPU
parallelism, cache-blocked GEMM and dense/CSR/ELLPACK matrix-vector kernels.
The server exposes `/health`, `/v1/capabilities` and `/v1/execute`.

The web UI uses the same benchmark/comparison/charts/history/methodology/environment
pages in both worlds, while showing only backends belonging to the selected world.
The server world can query CPU/GPU capabilities and exposes a GPU preference. GPU
kernels are kept separate from the CPU core so their transfer cost and device
characteristics can be benchmarked independently rather than being hidden by a
CPU fallback.

## v0.5.0 — Numerical operation expansion

TRACE now has a single operation catalog shared by the numerical core, the browser benchmark and the native server. The sidebar groups operations into compact tabs so the fixed viewport remains usable as the library grows.

### Implemented in the own Rust core

**BLAS-like Level 1**
- vector addition
- vector scaling
- dot product
- Euclidean norm
- AXPY
- 3D cross product

**Level 2 / Level 3**
- GEMV
- matrix-vector multiplication
- triangular solve
- GER rank-1 update
- GEMM
- TRSM
- SYRK

**Matrix operations**
- transpose
- inverse through LU solves
- determinant through LU
- trace
- Frobenius norm
- 1-norm
- infinity norm
- spectral norm through power iteration
- Kronecker product
- `Ax=b`
- `AX=B`

**Factorizations**
- LU with partial pivoting
- Cholesky
- LDLᵀ
- Householder QR
- Jacobi one-sided SVD

**Spectral reductions**
- symmetric eigenvalue/eigenvector decomposition
- Hessenberg reduction

The same kernels are exposed through the native server and the Rust/WebAssembly binding layer. Browser WASM is loaded from `/wasm/trace_numerical_wasm.js` after running `npm run wasm:build`.

### Explicitly not fabricated

The catalog also reserves slots for Bunch–Kaufman, rank-revealing factorization, polar decomposition, general eigenvalue/Schur, Jordan, Frobenius, Hermite, Smith and Popov forms. These entries remain marked **planned** until their production kernels and the required numerical domains (especially exact integer/polynomial arithmetic) are implemented. The benchmark cannot execute or report them as successful operations while they are planned.

### Design rule

TRACE is the implementation under test. OpenBLAS, BLIS, oneMKL, Eigen, cuBLAS, rocBLAS and other libraries are reference competitors for later adapters and benchmarking only; they are not dependencies of the own numerical algorithms.
