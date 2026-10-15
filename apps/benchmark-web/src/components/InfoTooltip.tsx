/**
 * Tooltip reutilizable para explicar métricas y valores numéricos.
 *
 * Se usa tanto en cabeceras como en celdas para que el significado y la
 * fórmula de cada dato estén disponibles sin saturar visualmente la tabla.
 */

interface InfoTooltipProps {
  readonly text: string;
}

/**
 * Muestra un indicador contextual que abre una explicación al pasar el cursor.
 */
export function InfoTooltip({ text }: InfoTooltipProps) {
  return (
    <span className="info-tooltip" tabIndex={0} aria-label={text}>
      <span aria-hidden="true">?</span>
      <span className="info-tooltip-popover" role="tooltip">{text}</span>
    </span>
  );
}

interface NumericValueProps {
  readonly value: string;
  readonly explanation: string;
  readonly className?: string;
}

/**
 * Renderiza un número con explicación accesible y visible al pasar el cursor.
 */
export function NumericValue({ value, explanation, className = "" }: NumericValueProps) {
  return (
    <span className={`numeric-with-help ${className}`}>
      <span title={explanation}>{value}</span>
      <InfoTooltip text={explanation} />
    </span>
  );
}
