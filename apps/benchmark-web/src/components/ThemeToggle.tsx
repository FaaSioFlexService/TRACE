/**
 * Selector de tema visual del laboratorio.
 *
 * El componente solo expone la interacción; la persistencia y el estado global
 * permanecen en App.tsx para que el resto de componentes no dependa del tema.
 */

import { useI18n } from "../i18n";

interface ThemeToggleProps {
  readonly theme: "dark" | "light";
  readonly onChange: (theme: "dark" | "light") => void;
}

/**
 * Alterna entre el tema oscuro y el tema claro.
 */
export function ThemeToggle({ theme, onChange }: ThemeToggleProps) {
  const { t } = useI18n();
  const nextTheme = theme === "dark" ? "light" : "dark";

  return (
    <button
      className="theme-toggle"
      type="button"
      onClick={() => onChange(nextTheme)}
      aria-label={t(nextTheme === "light" ? "Switch to light theme" : "Switch to dark theme")}
      title={t(nextTheme === "light" ? "Switch to light theme" : "Switch to dark theme")}
    >
      <span aria-hidden="true">{theme === "dark" ? "☀" : "☾"}</span>
      <span>{theme === "dark" ? t("Light") : t("Dark")}</span>
    </button>
  );
}
