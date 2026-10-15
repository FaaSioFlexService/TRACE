/**
 * Navegación por solapas reutilizable.
 *
 * El componente no conoce qué contenido representa cada tab; únicamente
 * gestiona selección, accesibilidad y presentación de la navegación.
 */

import type { ReactNode } from "react";
import { useI18n } from "../i18n";

export interface TabItem {
  readonly id: string;
  readonly label: string;
  readonly badge?: string;
  readonly disabled?: boolean;
}

interface TabsProps {
  readonly items: readonly TabItem[];
  readonly activeId: string;
  readonly onChange: (id: string) => void;
  readonly children?: ReactNode;
}

/**
 * Renderiza una barra de tabs con estado activo persistente.
 */
export function Tabs({ items, activeId, onChange }: TabsProps) {
  const { t } = useI18n();
  return (
    <div className="tabs" role="tablist" aria-label={t("Benchmark sections")}>
      {items.map((item) => (
        <button
          key={item.id}
          type="button"
          role="tab"
          aria-selected={item.id === activeId}
          disabled={item.disabled}
          className={item.id === activeId ? "tab active" : "tab"}
          onClick={() => onChange(item.id)}
        >
          <span>{item.label}</span>
          {item.badge ? <span className="tab-badge">{item.badge}</span> : null}
        </button>
      ))}
    </div>
  );
}
