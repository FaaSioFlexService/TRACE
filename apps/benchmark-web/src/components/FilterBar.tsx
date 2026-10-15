/**
 * Barra de filtros del histórico.
 *
 * El componente solo representa controles; la lógica de filtrado permanece en
 * la página para que pueda reutilizarse con diferentes fuentes de datos.
 */

interface FilterBarProps {
  readonly operation: string;
  readonly backend: string;
  readonly precision: string;
  readonly format: string;
  readonly dateFrom: string;
  readonly dateTo: string;
  readonly operations: string[];
  readonly backends: string[];
  readonly onOperationChange: (value: string) => void;
  readonly onBackendChange: (value: string) => void;
  readonly onPrecisionChange: (value: string) => void;
  readonly onFormatChange: (value: string) => void;
  readonly onDateFromChange: (value: string) => void;
  readonly onDateToChange: (value: string) => void;
  readonly onClear: () => void;
}

/**
 * Renderiza los filtros disponibles para el histórico.
 */
export function FilterBar(props: FilterBarProps) {
  return (
    <div className="filter-bar">
      <label>Operation<select value={props.operation} onChange={(event) => props.onOperationChange(event.target.value)}>
        <option value="all">All</option>
        {props.operations.map((operation) => <option key={operation} value={operation}>{operation}</option>)}
      </select></label>
      <label>Backend<select value={props.backend} onChange={(event) => props.onBackendChange(event.target.value)}>
        <option value="all">All</option>
        {props.backends.map((backend) => <option key={backend} value={backend}>{backend}</option>)}
      </select></label>
      <label>Precision<select value={props.precision} onChange={(event) => props.onPrecisionChange(event.target.value)}>
        <option value="all">All</option><option value="float32">FLOAT32</option><option value="float64">FLOAT64</option>
      </select></label>
      <label>Format<select value={props.format} onChange={(event) => props.onFormatChange(event.target.value)}>
        <option value="all">All</option><option value="dense">Dense</option><option value="csr">CSR</option><option value="csc">CSC</option><option value="ellpack">ELLPACK</option><option value="none">None</option>
      </select></label>
      <label>From<input type="date" value={props.dateFrom} onChange={(event) => props.onDateFromChange(event.target.value)} /></label>
      <label>To<input type="date" value={props.dateTo} onChange={(event) => props.onDateToChange(event.target.value)} /></label>
      <button className="secondary-button" type="button" onClick={props.onClear}>Clear filters</button>
    </div>
  );
}
