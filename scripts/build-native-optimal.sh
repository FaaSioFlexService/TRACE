#!/usr/bin/env bash
# Construye el servidor nativo optimizando el codigo para la CPU anfitriona.
# Este modo debe utilizarse solo cuando el binario no se vaya a mover a otra
# maquina con un conjunto de instrucciones inferior.
set -euo pipefail

export RUSTFLAGS="${RUSTFLAGS:-} -C target-cpu=native"
cargo build --workspace --release
