/**
 * Selector de backends/librerías para una ejecución.
 *
 * Las opciones representan backends realmente registrados. La UI no permite
 * seleccionar una librería que no tenga un adaptador implementado.
 */

import type { BenchmarkBackendDefinition, BenchmarkBackendId } from "../benchmark";
import { OPERATION_CATALOG, type MatrixFormat, type OperationName } from "@trace/vector-matrix-core";
import { useI18n } from "../i18n";

interface BackendSelectorProps {
  readonly backends: readonly BenchmarkBackendDefinition[];
  readonly selected: readonly BenchmarkBackendId[];
  readonly operation: OperationName;
  readonly matrixFormats: readonly MatrixFormat[];
  readonly onChange: (selected: BenchmarkBackendId[]) => void;
}

/**
 * Alterna librerías conservando el orden declarado en el registry.
 */
export function BackendSelector({ backends, selected, operation, matrixFormats, onChange }: BackendSelectorProps) {
  const { t } = useI18n();
  function toggle(id: BenchmarkBackendId): void {
    const next = selected.includes(id)
      ? selected.filter((current) => current !== id)
      : [...selected, id];
    onChange(next);
  }

  return (
    <fieldset>
      <legend>{t("Benchmark backends")}</legend>
      <div className="backend-list">
        {backends.map((backend) => {
          const supportsOperation = backend.operations.includes(operation);
          const operationSpec = OPERATION_CATALOG.find((spec) => spec.id === operation);
          const matrixOperation = operationSpec?.category !== "level1";
          const supportsFormat = matrixOperation
            ? matrixFormats.some((format) => backend.matrixFormats.includes(format))
            : true;
          const available = supportsOperation && supportsFormat && backend.available !== false;

          return (
            <label className={available ? "backend-option" : "backend-option disabled"} key={backend.id}>
              <input type="checkbox" checked={selected.includes(backend.id)} disabled={!available} onChange={() => toggle(backend.id)} />
              <span className="backend-option-main">
                <strong>{backend.name}</strong>
                <small>{backend.version} · {available ? backend.environment : (backend.available === false ? t("not linked or detected on this execution world") : t("not compatible with selected operation/format"))}</small>
              </span>
            </label>
          );
        })}
      </div>
      <p className="field-help">{t("Only selected libraries are stored in this test bank. The comparison cannot invent results that were never executed.")}</p>
    </fieldset>
  );
}
