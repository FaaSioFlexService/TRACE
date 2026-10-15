/**
 * Tabla genérica reutilizable.
 *
 * Ordenación y selección pertenecen al componente porque son capacidades
 * propias de una tabla; la paginación se mantiene deliberadamente fuera.
 */

import { useMemo, useState, type ReactNode } from "react";

export type SortDirection = "none" | "asc" | "desc";

export interface TableColumn<T> {
  readonly key: string;
  readonly header: ReactNode;
  readonly sortable?: boolean;
  readonly render: (row: T) => ReactNode;
  readonly getSortValue?: (row: T) => string | number | boolean | null;
}

interface DataTableProps<T extends { id: string }> {
  readonly rows: T[];
  readonly columns: TableColumn<T>[];
  readonly selectable?: boolean;
  readonly selectedIds?: Set<string>;
  readonly onSelectionChange?: (ids: Set<string>) => void;
  readonly emptyMessage?: string;
}

/**
 * Tabla estándar con ordenación de tres estados y selección opcional.
 */
export function DataTable<T extends { id: string }>({
  rows,
  columns,
  selectable = false,
  selectedIds = new Set<string>(),
  onSelectionChange,
  emptyMessage = "No records found.",
}: DataTableProps<T>) {
  const [sortKey, setSortKey] = useState<string | null>(null);
  const [sortDirection, setSortDirection] = useState<SortDirection>("none");

  const orderedRows = useMemo(() => {
    if (!sortKey || sortDirection === "none") return rows;

    const column = columns.find((candidate) => candidate.key === sortKey);
    if (!column?.getSortValue) return rows;

    return [...rows].sort((left, right) => {
      const leftValue = column.getSortValue!(left);
      const rightValue = column.getSortValue!(right);

      if (leftValue === rightValue) return 0;
      if (leftValue === null) return 1;
      if (rightValue === null) return -1;

      const comparison = leftValue < rightValue ? -1 : 1;
      return sortDirection === "asc" ? comparison : -comparison;
    });
  }, [columns, rows, sortDirection, sortKey]);

  /**
   * Avanza por none -> asc -> desc -> none.
   */
  function handleSort(column: TableColumn<T>): void {
    if (!column.sortable) return;

    if (sortKey !== column.key) {
      setSortKey(column.key);
      setSortDirection("asc");
      return;
    }

    if (sortDirection === "none") setSortDirection("asc");
    else if (sortDirection === "asc") setSortDirection("desc");
    else {
      setSortKey(null);
      setSortDirection("none");
    }
  }

  /**
   * Cambia la selección de una fila sin modificar la colección original.
   */
  function toggleRow(id: string): void {
    const next = new Set(selectedIds);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    onSelectionChange?.(next);
  }

  /**
   * Selecciona o deselecciona todas las filas de la vista filtrada/ordenada.
   */
  function toggleAll(): void {
    const next = new Set(selectedIds);
    const allSelected = orderedRows.length > 0 && orderedRows.every((row) => next.has(row.id));

    for (const row of orderedRows) {
      if (allSelected) next.delete(row.id);
      else next.add(row.id);
    }

    onSelectionChange?.(next);
  }

  return (
    <div className="standard-table-wrap">
      <table className="standard-table">
        <thead>
          <tr>
            {selectable && (
              <th className="selection-column">
                <input
                  aria-label="Select all visible rows"
                  type="checkbox"
                  checked={orderedRows.length > 0 && orderedRows.every((row) => selectedIds.has(row.id))}
                  onChange={toggleAll}
                />
              </th>
            )}
            {columns.map((column) => {
              const active = sortKey === column.key && sortDirection !== "none";
              const marker = active ? (sortDirection === "asc" ? "↑" : "↓") : "↕";
              return (
                <th key={column.key} scope="col">
                  {column.sortable ? (
                    <button className="sort-button" type="button" onClick={() => handleSort(column)}>
                      <span>{column.header}</span>
                      <span aria-hidden="true" className={active ? "sort-marker active" : "sort-marker"}>
                        {marker}
                      </span>
                    </button>
                  ) : (
                    column.header
                  )}
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody>
          {orderedRows.map((row) => (
            <tr key={row.id}>
              {selectable && (
                <td className="selection-column">
                  <input
                    aria-label={`Select ${row.id}`}
                    type="checkbox"
                    checked={selectedIds.has(row.id)}
                    onChange={() => toggleRow(row.id)}
                  />
                </td>
              )}
              {columns.map((column) => <td key={column.key}>{column.render(row)}</td>)}
            </tr>
          ))}
          {orderedRows.length === 0 && (
            <tr>
              <td colSpan={columns.length + (selectable ? 1 : 0)} className="empty-state">
                {emptyMessage}
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
