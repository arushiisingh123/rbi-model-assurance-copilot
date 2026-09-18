/**
 * Recharts wrappers used by the explainability, fairness and monitoring pages.
 *
 * Two rules these charts follow:
 *
 * 1. THE AXIS IS ALWAYS LABELLED WITH THE BACKEND'S SCALE. A contribution
 *    chart without its unit invites the reader to compare log-odds against
 *    probabilities, which is the single easiest way to misread this data.
 * 2. Signed values keep their sign and their colour. Sorting by magnitude is
 *    a display choice; flipping a negative contribution to positive would
 *    reverse what the model actually did.
 */
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { humanise, num } from "../utils/format";

const POSITIVE = "#b91c1c"; // pushes toward the unfavourable class (BAD)
const NEGATIVE = "#0369a1"; // pushes toward the favourable class (GOOD)
const NEUTRAL = "#1e3a5f";

function ChartTooltip({ active, payload, unit }) {
  if (!active || !payload?.length) return null;
  const point = payload[0].payload;
  return (
    <div className="bg-base-100 border border-base-300 rounded shadow-sm px-3 py-2 text-xs">
      <p className="font-medium">{humanise(point.feature ?? point.label)}</p>
      <p className="text-base-content/70 font-mono mt-0.5">
        {num(point.value, 6)}
        {unit ? ` ${unit}` : ""}
      </p>
    </div>
  );
}

/**
 * Horizontal signed-contribution chart (SHAP / LIME / feature importance).
 *
 * `unit` must be the backend's own scale string. It is required rather than
 * optional on purpose -- an unlabelled contribution axis is a correctness
 * problem, not a cosmetic one.
 */
export function ContributionChart({ rows, unit, signed = true, height }) {
  if (!rows?.length) return null;
  const chartHeight = height ?? Math.max(220, rows.length * 26 + 40);

  return (
    <div>
      <ResponsiveContainer width="100%" height={chartHeight}>
        <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 24, bottom: 4, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" horizontal={false} />
          <XAxis
            type="number"
            tick={{ fontSize: 11, fill: "#6b7280" }}
            stroke="#d1d5db"
          />
          <YAxis
            type="category"
            dataKey="feature"
            width={190}
            tick={{ fontSize: 11, fill: "#374151" }}
            tickFormatter={humanise}
            stroke="#d1d5db"
          />
          <Tooltip content={<ChartTooltip unit={unit} />} />
          <Bar dataKey="value" radius={[0, 2, 2, 0]} isAnimationActive={false}>
            {rows.map((row, index) => (
              <Cell
                key={index}
                fill={signed ? (row.value >= 0 ? POSITIVE : NEGATIVE) : NEUTRAL}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <p className="mt-2 text-xs text-base-content/60">
        Horizontal axis: contribution on the <strong>{unit}</strong> scale, as
        reported by the backend.
        {signed && (
          <>
            {" "}
            <span style={{ color: POSITIVE }}>Red</span> pushes toward the
            unfavourable class, <span style={{ color: NEGATIVE }}>blue</span>{" "}
            toward the favourable one.
          </>
        )}
      </p>
    </div>
  );
}

/** Grouped bar chart for per-feature PSI / KS drift metrics. */
export function DriftChart({ perFeature, height = 300 }) {
  if (!perFeature?.length) return null;
  const rows = perFeature.map((row) => ({
    feature: row.feature,
    PSI: Number(row.psi),
    KS: Number(row.ks_statistic),
  }));

  return (
    <div>
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={rows} margin={{ top: 4, right: 16, bottom: 60, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis
            dataKey="feature"
            angle={-40}
            textAnchor="end"
            interval={0}
            height={70}
            tick={{ fontSize: 10, fill: "#374151" }}
            tickFormatter={humanise}
            stroke="#d1d5db"
          />
          <YAxis tick={{ fontSize: 11, fill: "#6b7280" }} stroke="#d1d5db" />
          <Tooltip
            contentStyle={{ fontSize: 12, borderRadius: 4, borderColor: "#e5e7eb" }}
            formatter={(value) => num(value, 4)}
          />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <Bar dataKey="PSI" fill={NEUTRAL} isAnimationActive={false} />
          <Bar dataKey="KS" fill="#0f766e" isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
      <p className="mt-2 text-xs text-base-content/60">
        Per-feature drift metrics. These are input-distribution measures; the
        PASS/WARNING/FAIL status comes from the backend&apos;s own thresholds,
        not from this chart.
      </p>
    </div>
  );
}

/** Per-group selection-rate chart for fairness. */
export function GroupRateChart({ groups, height = 260 }) {
  if (!groups?.length) return null;
  const rows = groups.map((group) => ({
    feature: group.group,
    value: Number(group.selection_rate),
  }));

  return (
    <div>
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={rows} margin={{ top: 4, right: 16, bottom: 50, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis
            dataKey="feature"
            angle={-25}
            textAnchor="end"
            interval={0}
            height={60}
            tick={{ fontSize: 10, fill: "#374151" }}
            stroke="#d1d5db"
          />
          <YAxis
            domain={[0, 1]}
            tick={{ fontSize: 11, fill: "#6b7280" }}
            stroke="#d1d5db"
          />
          <Tooltip
            contentStyle={{ fontSize: 12, borderRadius: 4, borderColor: "#e5e7eb" }}
            formatter={(value) => num(value, 4)}
          />
          <Bar dataKey="value" fill={NEUTRAL} radius={[2, 2, 0, 0]} isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
      <p className="mt-2 text-xs text-base-content/60">
        Selection rate per protected group, as calculated by the fairness
        module.
      </p>
    </div>
  );
}
