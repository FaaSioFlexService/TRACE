/**
 * Subsolapas de ejecuciones históricas.
 *
 * Cada banco de pruebas conserva sus propios backends seleccionados. Esto
 * evita mezclar resultados de dos campañas distintas en una comparación.
 */

import type { BenchmarkRun } from "../benchmark";
import { useI18n } from "../i18n";

interface RunTabsProps {
  readonly runs: readonly BenchmarkRun[];
  readonly activeRunId: string | null;
  readonly onChange: (runId: string) => void;
}

/**
 * Renderiza una navegación compacta por campañas históricas.
 */
export function RunTabs({ runs, activeRunId, onChange }: RunTabsProps) {
  const { t } = useI18n();
  return (
    <div className="run-tabs" role="tablist" aria-label={t("Benchmark runs")}>
      {runs.map((run) => (
        <button
          key={run.id}
          type="button"
          role="tab"
          aria-selected={run.id === activeRunId}
          className={run.id === activeRunId ? "run-tab active" : "run-tab"}
          onClick={() => onChange(run.id)}
        >
          <strong>{run.configuration.operation}</strong>
          <span>{new Date(run.startedAt).toLocaleString()}</span>
          <small>{run.selectedBackends.length} backends · {run.caseCount} {t("cases")}</small>
        </button>
      ))}
    </div>
  );
}
