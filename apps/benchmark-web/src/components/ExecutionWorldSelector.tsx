/**
 * Selector del mundo de ejecucion.
 *
 * Browser mantiene la experiencia offline/WASM. Server cambia el catalogo a
 * backends nativos y deja visible la politica CPU/GPU elegida para el servidor.
 */

import { useI18n } from "../i18n";

export type ExecutionWorld = "browser" | "server";
export type ServerExecutionPolicy = "cpu" | "gpu";

interface ExecutionWorldSelectorProps {
  readonly world: ExecutionWorld;
  readonly serverUrl: string;
  readonly serverExecutionPolicy: ServerExecutionPolicy;
  readonly onWorldChange: (world: ExecutionWorld) => void;
  readonly onServerUrlChange: (url: string) => void;
  readonly onServerExecutionPolicyChange: (policy: ServerExecutionPolicy) => void;
}

/**
 * Renderiza el selector de entorno y la politica de aceleracion del servidor.
 */
export function ExecutionWorldSelector({
  world,
  serverUrl,
  serverExecutionPolicy,
  onWorldChange,
  onServerUrlChange,
  onServerExecutionPolicyChange,
}: ExecutionWorldSelectorProps) {
  const { t } = useI18n();

  return (
    <fieldset className="execution-world-panel">
      <legend>{t("Execution world")}</legend>
      <div className="segmented-control" role="group" aria-label={t("Execution world")}>
        <button type="button" className={world === "browser" ? "active" : ""} onClick={() => onWorldChange("browser")}>
          {t("Browser / Offline")}
        </button>
        <button type="button" className={world === "server" ? "active" : ""} onClick={() => onWorldChange("server")}>
          {t("Server / Native")}
        </button>
      </div>
      {world === "server" ? (
        <div className="server-settings">
          <label>
            {t("Server URL")}
            <input value={serverUrl} onChange={(event) => onServerUrlChange(event.target.value)} placeholder="http://localhost:8080" />
          </label>
          <label>
            {t("Hardware policy")}
            <select value={serverExecutionPolicy} onChange={(event) => onServerExecutionPolicyChange(event.target.value as ServerExecutionPolicy)}>
              <option value="cpu">{t("CPU only / automatic parallelism")}</option>
              <option value="gpu">{t("Use GPU when available")}</option>
            </select>
          </label>
          <p className="field-help">{t("The server detects CPU features and GPU adapters. GPU execution is selected explicitly so transfer overhead can be measured separately.")}</p>
        </div>
      ) : null}
    </fieldset>
  );
}
