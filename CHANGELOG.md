# v0.3.4

## Clarity, metric explainability and internationalization

- Clarified the comparison table semantics: P50 is explicitly documented as the 50th percentile/median, not the operation name.
- Renamed the ratio column to **Relative time (Library / TRACE)**.
- Added hover/focus tooltips for P50, P95, relative time, GFLOP/s, CV, confidence intervals, working set and chart values.
- Added chart point and axis explanations.
- Added a persistent ES/EN language switch in the top-right corner and centralized UI translations.
- Kept technical identifiers and backend names unchanged so benchmark data remains auditable.
- Kept the fixed application shell and external table pagination.

# Changelog

## 0.3.3

- Added a historical campaign selector to the first Benchmark tab; the newest completed campaign is selected by default.
- Added an explicit external-adapter contract for browser numerical libraries.
- Added adapter upload, template generation and session persistence.
- Uploaded adapters execute only inside the Benchmark Worker.
- Added campaign-level charts for runtime, GFLOP/s, speedup, working-set size, coefficient of variation, P95/P50 tail ratio and P50 confidence intervals.
- Added chart sub-views so the professional report remains readable without page scrolling.
- Added stable colors for dynamically uploaded backends.
- Kept comparison visibility campaign-scoped: a backend must have actually executed in the selected campaign.
- Expanded methodology and auditability information.

## 0.3.2

- Moved benchmark execution into a Web Worker so long test banks no longer block the browser main thread.
- Added explicit run cancellation without persisting partial campaigns.
- Increased UI typography and contrast for readable desktop presentation.
- Kept the configuration sidebar fixed and visible without scrolling.
- Added clearer execution-state controls and backend identity.

## 0.3.1

- Rediseño compacto del shell para eliminar el scroll de página y el scroll interno de la configuración.
- Sidebar de configuración fija y optimizada para mostrar operación, formatos, tamaños, densidades y backends en un único viewport.
- Paginación externa para resultados propios, comparación, historial y previsualización del banco.
- Tabla comparativa normalizada a una fila por caso/backend para mejorar lectura y distinguir TRACE de las referencias.
- Colores persistentes por backend y etiqueta explícita `OUR LIBRARY` para TRACE Matrix Core.
- Selector explícito de campaña y de batería/slice de pruebas en Charts.
- Leyendas de gráficos con identificación inequívoca de TRACE y competidores.
- Tema claro/oscuro persistente mediante `localStorage`.
- Mejora visual general de contraste, jerarquía y densidad de información.

## 0.3.0 — Benchmark Workbench

### Added

- Fixed application viewport with internal scrolling instead of browser-page scrolling.
- Persistent sidebar with independent configuration and benchmark execution controls.
- Reusable tab navigation and campaign sub-tabs.
- Separate **My Library** tab containing only TRACE Matrix Core results.
- Separate **Comparison** tab with historical benchmark campaigns.
- Competitor checkboxes restricted to backends actually executed in the selected campaign.
- Versioned IndexedDB model with independent benchmark runs and result records.
- Browser benchmark backend registry.
- JavaScript baseline backend.
- `ml-matrix` browser adapter for common dense operations.
- Runtime calibration with inner iterations to avoid sub-millisecond measurements collapsing to zero.
- P50, P90, P95, P99, IQR, standard deviation, coefficient of variation and approximate 95% confidence interval.
- Actual NNZ and actual sparse density recorded for each matrix case.
- ELLPACK storage slots and padding ratio recorded for sparse analysis.
- SVG runtime, GFLOP/s and relative-performance charts.
- Environment capture per benchmark campaign.
- Historical campaign tabs prevent comparing libraries that were not executed in that campaign.

### Changed

- Benchmark metadata was removed from the independent numerical core package.
- Benchmark-specific types now live in the web application's benchmark engine.
- Result IDs include run, backend and case, preventing collisions between campaigns.
- Comparison is performed per mathematical case rather than by mixing unrelated runs.

### Methodology

- Data generation and representation conversion remain outside timed regions.
- Every selected backend receives the same logical problem and deterministic seed.
- Unsupported backend/format combinations are recorded as `skipped`, not fabricated as measurements.

## 0.4.0 - Native HPC execution architecture

- Added a Rust workspace for the native numerical core.
- Added CPU runtime detection for architecture, logical threads and SIMD features.
- Added runtime CPU planning with scalar/SIMD selection and automatic parallelism.
- Added cache-blocked dense GEMM and row-partitioned dense/CSR/ELLPACK matrix-vector kernels.
- Added explicit AVX2/FMA kernels for vector addition, scaling, dot product and dense GEMV.
- Added native benchmark server endpoints for health, capabilities and execution.
- Added a GPU detection crate based on wgpu behind an optional native-gpu feature.
- Added a native backend registry for OpenBLAS, BLIS, oneMKL, AOCL, Eigen, Arm PL, Accelerate, cuBLAS, rocBLAS and NVPL. Backends remain unavailable until a real native adapter is linked.
- Added Browser/Offline vs Server/Native execution worlds in the web UI.
- Added CPU-only vs GPU-preferred server policy and server capability detection.
- Replaced the architectural assumption that browser libraries are the only comparison universe.
- Campaign history remains isolated by execution world.

## 0.5.0 — Numerical library expansion

- Added a centralized operation catalog covering BLAS-like Level 1/2/3, matrix utilities, factorizations, spectral algorithms and exact-domain roadmap operations.
- Added Rust-native AXPY, cross product, GEMV, triangular solve, GER, TRSM, SYRK, transpose, matrix norms, Kronecker product, determinant, inverse, linear-system solves and multiple-RHS solves.
- Added CSC sparse storage and SpMV alongside dense, CSR and ELLPACK.
- Added own Rust LU with partial pivoting, Cholesky, LDLᵀ, Householder QR, rank-revealing QR, Jacobi one-sided SVD, polar decomposition, symmetric eigen decomposition and Hessenberg reduction.
- Added matching WebAssembly exports for the implemented operations, including CSR/CSC/ELLPACK SpMV.
- Added AVX-512F dispatch in the own vector kernels and cache-aware GEMM panel packing with SIMD dot micro-kernels.
- Extended the native benchmark server to execute the implemented operation catalog and return flattened auditable results/checksums.
- Added a compact operation navigator in the fixed sidebar, grouped into Level 1, Level 2, Level 3, matrix, factorization, eigen/spectral and exact-domain tabs.
- Added operation formulas and complexity labels to the UI.
- Planned exact/specialized decompositions remain explicitly disabled instead of being reported as successful benchmarks.
- Added a browser WASM loading path and build output contract under `apps/benchmark-web/public/wasm`.
