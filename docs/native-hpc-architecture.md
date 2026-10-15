# Arquitectura HPC nativa de TRACE

## Dos mundos de ejecucion

TRACE HPC Benchmark Lab separa explicitamente:

1. **Browser / Offline**: JavaScript y WebAssembly. No requiere servidor.
2. **Server / Native**: Rust nativo y adapters HPC del sistema. El frontend usa
   el servidor como motor de ejecucion.

Las paginas de Benchmark, Comparison, Charts, History, Methodology y Environment
son las mismas, pero el catalogo de backends se filtra por el mundo de ejecucion.

## CPU

El core Rust detecta arquitectura y caracteristicas SIMD en runtime. El plan CPU
elige entre scalar y SIMD y decide si el tamaño justifica paralelismo. El reparto de
trabajo usa Rayon como runtime de data-parallelism para evitar crear y destruir
hilos por cada operacion. Para GEMM
se usa blocking para mejorar localidad de cache; para GEMV y sparse matrix-vector
se divide por filas, que es el acceso natural del layout row-major/CSR/ELLPACK.

Rust permite generar codigo para `target-cpu=native` cuando el binario se construye
para una maquina conocida y tambien permite dispatch dinamico por caracteristicas
cuando se necesita un binario portable. La estrategia adoptada por TRACE debe
mantener un fallback seguro y medir el beneficio real de cada variante.

## GPU

El servidor detecta adaptadores GPU mediante `wgpu` cuando se compila con el
backend GPU. La politica de usuario tiene dos estados: CPU-only y Use GPU when
available. La ejecucion GPU debe registrar dispositivo, backend, driver y costes
de transferencia por separado.

La implementacion de kernels GPU se mantiene como modulo independiente porque
dense, CSR y ELLPACK requieren estrategias de acceso distintas. No se presenta
un fallback CPU como resultado GPU: si el kernel GPU no esta habilitado, el
benchmark debe marcarse como no disponible.

## Bibliotecas nativas

El registry contempla adapters para OpenBLAS, BLIS, Intel oneMKL, AMD AOCL, Eigen,
Arm Performance Libraries, Apple Accelerate, NVIDIA cuBLAS, AMD rocBLAS y NVIDIA
Performance Libraries. Solo un adapter realmente enlazado puede pasar a estado
`available`.

Esto permite mantener una comparativa cientificamente valida: no se comparan
implementaciones que en realidad no se ejecutaron.

## Build portable vs build optimizado para la maquina

Hay dos estrategias y no deben confundirse:

- **Portable**: `cargo build --release`. El runtime selecciona kernels seguros
  con deteccion dinamica de features.
- **Maximum local performance**: `scripts/build-native-optimal.sh`, que usa
  `-C target-cpu=native`. Este binario debe ejecutarse en una maquina compatible
  con la CPU para la que se construyo.

Rust documenta tanto `target-cpu=native` como el dispatch dinamico mediante
`is_x86_feature_detected!`/`is_aarch64_feature_detected!`. La segunda estrategia
es la adecuada cuando queremos distribuir un unico binario portable; la primera
es la adecuada para un runner dedicado de benchmark.
