# TRACE Rust WASM runtime

Run `npm run wasm:build` from the repository root to generate:

- `trace_numerical_wasm.js`
- `trace_numerical_wasm_bg.wasm`
- related wasm-bindgen metadata

The benchmark web app probes this directory before enabling the TRACE Rust WASM backend.
