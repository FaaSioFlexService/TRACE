# Professional benchmarking protocol

## Benchmark identity

Every result is associated with:

- campaign ID and timestamp;
- operation;
- scalar type;
- dimensions;
- matrix format;
- requested and actual density;
- NNZ;
- backend and library version;
- execution environment;
- warmup count;
- measured sample count;
- inner iteration count.

## Timing

Preparation is excluded from the measured region. The engine calibrates inner iterations to obtain a stable wall-clock signal and performs a separate warmup phase.

For production TFM runs, the target sample time and sample count should be fixed and reported as part of the methodology.

## Statistics

Report at minimum:

- P50;
- P90;
- P95;
- P99;
- mean;
- standard deviation;
- coefficient of variation;
- IQR;
- approximate 95% confidence interval.

The UI additionally exposes P95/P50 tail ratio as a compact measure of latency instability.

## Comparison rules

A comparison is valid only when operation, precision, dimensions, format and requested density match and all compared backends belong to the same campaign.

Browser and native/HPC measurements are separate benchmark planes.

## Recommended report views

1. Runtime vs problem size.
2. GFLOP/s vs problem size.
3. Relative runtime/speedup.
4. Working-set size.
5. Coefficient of variation.
6. P95/P50 tail ratio.
7. P50 with confidence interval.
8. Environment/context.
9. Raw result export.

The laboratory must not reduce these measurements to a single arbitrary score unless a justified domain-specific model is explicitly defined.
