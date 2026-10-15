# TRACE Benchmark Adapter API

## Purpose

An adapter is the compatibility layer between an external numerical library and the TRACE benchmark engine.

The benchmark engine owns:

- deterministic test-case generation;
- warmup;
- calibration;
- timing;
- statistics;
- campaign persistence;
- comparison and charts.

The adapter owns only library-specific setup and the actual numerical kernel.

## Bundle format

```json
{
  "metadata": {
    "id": "my-library",
    "name": "My Library",
    "version": "1.0.0",
    "environment": "browser",
    "description": "Adapter description",
    "operations": ["matrixVectorMultiply"],
    "matrixFormats": ["dense"]
  },
  "module": "export function createAdapter() { ... }"
}
```

## Module contract

The module must export:

```ts
export function createAdapter() {
  return {
    prepare({ testCase, inputs }) {
      // Prepare library objects here. This code is outside the timed region.
      return {
        execute() {
          // Only the numerical operation is timed.
          return 0;
        },
        densityActual: testCase.densityRequested,
        nnz: 0,
        ellpackStorageSlots: 0,
        ellpackPaddingRatio: 0,
        estimatedMemoryBytes: 0,
      };
    },
  };
}
```

## Canonical inputs

`inputs` contains deterministic operands generated from the benchmark case:

- `vector`;
- `otherVector` for vector addition;
- `leftMatrix`;
- `rightMatrix` for GEMM;
- `leftCsr`;
- `leftEllpack`.

The adapter must not regenerate a different workload inside `execute()`.

## Unsupported cases

Return `null` from `prepare()` when the library does not support a requested operation/format. The engine records a skipped result instead of fabricating a measurement.

## Security

Uploaded JavaScript is executed in the benchmark Worker. This prevents UI blocking but does not make arbitrary JavaScript trustworthy. Only trusted adapters should be uploaded.

A future WASM adapter contract will provide a stronger isolation model and a language-independent binary boundary.
