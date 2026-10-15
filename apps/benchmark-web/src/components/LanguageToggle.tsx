/**
 * Selector compacto de idioma situado en la esquina superior derecha.
 */

import { useI18n } from "../i18n";

/**
 * Permite cambiar entre español e inglés sin recargar la aplicación.
 */
export function LanguageToggle() {
  const { language, setLanguage } = useI18n();
  return (
    <div className="language-toggle" role="group" aria-label="Language / Idioma">
      <button type="button" className={language === "es" ? "active" : ""} onClick={() => setLanguage("es")}>ES</button>
      <button type="button" className={language === "en" ? "active" : ""} onClick={() => setLanguage("en")}>EN</button>
    </div>
  );
}
