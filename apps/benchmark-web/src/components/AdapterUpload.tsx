/**
 * Gestor de adaptadores de benchmark subidos por el usuario.
 *
 * El laboratorio no intenta adivinar la API de una librería arbitraria. En su
 * lugar acepta un contrato explícito y portable que puede ejecutar el Worker.
 * Esto mantiene el benchmark reproducible y evita generar llamadas ambiguas.
 */

import { useRef, useState, type CSSProperties } from "react";
import type { UploadedBenchmarkAdapterDefinition, OperationName } from "../benchmark";
import type { MatrixFormat } from "@trace/vector-matrix-core";
import { useI18n } from "../i18n";

interface AdapterUploadProps {
  readonly adapters: readonly UploadedBenchmarkAdapterDefinition[];
  readonly onAdd: (adapter: UploadedBenchmarkAdapterDefinition) => void;
  readonly onRemove: (adapterId: string) => void;
}

/**
 * Comprueba la forma mínima del bundle JSON sin ejecutar su código.
 */
function parseAdapterBundle(source: string): UploadedBenchmarkAdapterDefinition {
  const bundle = JSON.parse(source) as {
    metadata?: Partial<UploadedBenchmarkAdapterDefinition>;
    module?: string;
  };

  if (!bundle.metadata || typeof bundle.module !== "string") {
    throw new Error("El archivo debe contener metadata y module.");
  }

  const metadata = bundle.metadata;
  const requiredStrings = ["id", "name", "version", "description"] as const;
  for (const field of requiredStrings) {
    if (typeof metadata[field] !== "string" || !metadata[field]) {
      throw new Error(`Falta metadata.${field}.`);
    }
  }

  if (metadata.environment !== "browser") {
    throw new Error("Un adaptador subido debe declarar environment=browser.");
  }
  const validOperations: OperationName[] = ["vectorAdd", "vectorScale", "dotProduct", "vectorNorm2", "matrixVectorMultiply", "matrixMatrixMultiply"];
  const validFormats: MatrixFormat[] = ["dense", "csr", "csc", "ellpack"];
  if (!Array.isArray(metadata.operations) || metadata.operations.length === 0 || metadata.operations.some((operation) => !validOperations.includes(operation as OperationName))) {
    throw new Error("metadata.operations contiene una operación no soportada por el contrato.");
  }
  if (!Array.isArray(metadata.matrixFormats) || metadata.matrixFormats.some((format) => !validFormats.includes(format as MatrixFormat))) {
    throw new Error("metadata.matrixFormats contiene un formato no soportado por el contrato.");
  }

  return {
    id: metadata.id,
    name: metadata.name,
    version: metadata.version,
    environment: "browser",
    description: metadata.description,
    operations: metadata.operations,
    matrixFormats: metadata.matrixFormats,
    source: bundle.module,
  } as UploadedBenchmarkAdapterDefinition;
}

/**
 * Crea un bundle de ejemplo que sirve como contrato oficial del laboratorio.
 */
function createAdapterTemplate(): void {
  const bundle = {
    metadata: {
      id: "my-library",
      name: "My uploaded library",
      version: "1.0.0",
      environment: "browser",
      description: "Adapter for a browser numerical library.",
      operations: ["matrixVectorMultiply"],
      matrixFormats: ["dense"],
    },
    module: `export function createAdapter() {\n  return {\n    prepare({ testCase, inputs }) {\n      if (testCase.operation !== "matrixVectorMultiply" || !inputs.leftMatrix) return null;\n\n      const matrix = inputs.leftMatrix.data;\n      const vector = inputs.vector;\n      const rows = inputs.leftMatrix.rows;\n      const columns = inputs.leftMatrix.columns;\n\n      return {\n        execute() {\n          let first = 0;\n          for (let row = 0; row < rows; row += 1) {\n            let sum = 0;\n            const offset = row * columns;\n            for (let column = 0; column < columns; column += 1) {\n              sum += matrix[offset + column] * vector[column];\n            }\n            if (row === 0) first = sum;\n          }\n          return first;\n        },\n        densityActual: testCase.densityRequested,\n        nnz: 0,\n        ellpackStorageSlots: 0,\n        ellpackPaddingRatio: 0,\n        estimatedMemoryBytes: matrix.byteLength + vector.byteLength,\n      };\n    },\n  };\n}\n`,
  };

  const blob = new Blob([JSON.stringify(bundle, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "trace-benchmark-adapter-template.json";
  anchor.click();
  URL.revokeObjectURL(url);
}

/**
 * Panel compacto para registrar adaptadores externos sin mezclar el código de
 * la librería con el motor de medición.
 */
export function AdapterUpload({ adapters, onAdd, onRemove }: AdapterUploadProps) {
  const { t } = useI18n();
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [message, setMessage] = useState("");

  async function handleFile(file: File): Promise<void> {
    try {
      const definition = parseAdapterBundle(await file.text());
      if (adapters.some((adapter) => adapter.id === definition.id)) {
        throw new Error(`Ya existe un adaptador con id=${definition.id}.`);
      }
      onAdd(definition);
      setMessage(`${definition.name} ${definition.version} registrado.`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    }
  }

  return (
    <fieldset className="adapter-upload">
      <legend>{t("External library adapter")}</legend>
      <div className="adapter-upload-actions">
        <button className="secondary-button" type="button" onClick={() => inputRef.current?.click()}>
          {t("Upload adapter")}
        </button>
        <button className="secondary-button" type="button" onClick={createAdapterTemplate}>
          {t("Template")}
        </button>
        <input
          ref={inputRef}
          type="file"
          accept="application/json,.json"
          hidden
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) void handleFile(file);
            event.currentTarget.value = "";
          }}
        />
      </div>
      <p className="field-help visible-help">
        {t("An adapter is uploaded, not an arbitrary library. The module executes only inside the Benchmark Worker.")}
      </p>
      {adapters.map((adapter) => (
        <div className="uploaded-adapter" key={adapter.id}>
          <span className="library-badge" style={{ "--badge-color": "#8b5cf6" } as CSSProperties}>{adapter.name}</span>
          <button className="icon-button" type="button" onClick={() => onRemove(adapter.id)} aria-label={`Remove ${adapter.name}`}>×</button>
        </div>
      ))}
      {message ? <p className="field-help visible-help">{message}</p> : null}
    </fieldset>
  );
}
