/**
 * Componente de paginación reutilizable.
 *
 * Se mantiene fuera de DataTable para poder paginar listas, tarjetas o tablas
 * sin acoplar la navegación a una implementación visual concreta.
 */

import { useI18n } from "../i18n";

interface PaginationProps {
  readonly currentPage: number;
  readonly totalItems: number;
  readonly pageSize: number;
  readonly onPageChange: (page: number) => void;
}

/**
 * Renderiza controles de navegación y tamaño de página.
 */
export function Pagination({ currentPage, totalItems, pageSize, onPageChange }: PaginationProps) {
  const { t } = useI18n();
  const totalPages = Math.max(1, Math.ceil(totalItems / pageSize));
  const safePage = Math.min(Math.max(currentPage, 1), totalPages);
  const start = totalItems === 0 ? 0 : (safePage - 1) * pageSize + 1;
  const end = Math.min(safePage * pageSize, totalItems);

  return (
    <div className="pagination">
      <span className="pagination-summary">{start}–{end} {t("of")} {totalItems}</span>
      <div className="pagination-controls">
        <button type="button" disabled={safePage <= 1} onClick={() => onPageChange(safePage - 1)}>{t("Previous")}</button>
        <span>{t("Page")} {safePage} / {totalPages}</span>
        <button type="button" disabled={safePage >= totalPages} onClick={() => onPageChange(safePage + 1)}>{t("Next")}</button>
      </div>
    </div>
  );
}
