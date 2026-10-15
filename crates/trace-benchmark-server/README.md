# TRACE Benchmark Server

Servidor nativo para ejecutar el mismo modelo de operaciones fuera del navegador.

## Ejecucion

```bash
cargo run -p trace-benchmark-server --release
```

Endpoints:

- `GET /health`
- `GET /v1/capabilities`
- `POST /v1/execute`

El endpoint de capabilities devuelve arquitectura CPU, instrucciones SIMD, hilos,
adaptadores GPU y el catalogo de adapters nativos enlazados.

Las bibliotecas externas se registran como disponibles solo cuando existe un
adapter nativo real compilado para el servidor. No se deben presentar como
benchmarks ejecutables si la biblioteca no esta instalada/enlazada.
