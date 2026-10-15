/**
 * Tarjeta de métrica reutilizable para KPIs del laboratorio.
 */

interface MetricCardProps {
  readonly label: string;
  readonly value: string;
  readonly detail?: string;
  readonly explanation?: string;
}

/**
 * Muestra una métrica principal con contexto opcional.
 */
export function MetricCard({ label, value, detail, explanation }: MetricCardProps) {
  return (
    <article className="metric-card">
      <span>{label}</span>
      <strong title={explanation}>{value}</strong>
      {detail ? <small>{detail}</small> : null}
    </article>
  );
}
