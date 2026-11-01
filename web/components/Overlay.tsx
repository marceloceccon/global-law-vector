"use client";

/** The control panel: colour mode, country toggles, legend, counts, technical caption. */
import { useMemo } from "react";
import type { PointsPayload } from "@/lib/points";
import type { ColorMode } from "./PointCloud";

const fmt = (n: number) => n.toLocaleString("en-US");
const pretty = (s: string) => s.replace(/-/g, " ");

export default function Overlay({
  data,
  visible,
  enabled,
  mode,
  onToggle,
  onAll,
  onMode,
}: {
  data: PointsPayload;
  visible: Uint8Array;
  enabled: ReadonlySet<string>;
  mode: ColorMode;
  onToggle: (country: string) => void;
  onAll: (on: boolean) => void;
  onMode: (mode: ColorMode) => void;
}) {
  const { shown, perCountry, perDomain } = useMemo(() => {
    const perDomain = new Array<number>(data.domains.length).fill(0);
    const perCountry = new Array<number>(data.countries.length).fill(0);
    let shown = 0;
    for (let i = 0; i < data.sample; i++) {
      if (!visible[i]) continue;
      shown++;
      perDomain[data.domain[i]]++;
      perCountry[data.country[i]]++;
    }
    return { shown, perCountry, perDomain };
  }, [data, visible]);
  const shownCountries = data.countries.filter((c) => enabled.has(c.name)).length;

  const p = data.projection;
  const caption =
    p.method === "umap"
      ? `UMAP, 3 components, n_neighbors ${p.params.n_neighbors}, min_dist ${p.params.min_dist}, ` +
        `${p.params.metric} metric; fitted on ${fmt(p.fitSize)} of ${fmt(data.sample)} chunks sampled ` +
        `evenly by country, the rest placed by transform. Computed ${p.createdAt.slice(0, 10)}.`
      : `PCA (fallback), 3 components fitted on ${fmt(p.fitSize)} of ${fmt(data.sample)} chunks; ` +
        `run \`glv project\` for the UMAP layout.`;

  return (
    <aside className="panel" aria-label="controls">
      <h1>global-law-vector · embedding space</h1>
      <p className="sub">
        showing <span className="count">{fmt(shown)}</span> points from{" "}
        <span className="count">{shownCountries}</span> {shownCountries === 1 ? "country" : "countries"}
        <br />
        sample of {fmt(data.sample)} from {fmt(data.totalChunks)} indexed chunks
      </p>

      <section>
        <h2>colour by</h2>
        <div className="switch" role="group" aria-label="colour by">
          <button type="button" aria-pressed={mode === "country"} onClick={() => onMode("country")}>
            country
          </button>
          <button type="button" aria-pressed={mode === "domain"} onClick={() => onMode("domain")}>
            legal domain
          </button>
        </div>
      </section>

      <section>
        <h2>
          countries
          <span className="links">
            <button type="button" onClick={() => onAll(true)}>all</button>
            <button type="button" onClick={() => onAll(false)}>none</button>
          </span>
        </h2>
        {data.countries.map((c, i) => (
          <button
            key={c.name}
            type="button"
            className="toggle"
            aria-pressed={enabled.has(c.name)}
            onClick={() => onToggle(c.name)}
            title={`${fmt(c.total)} chunks indexed, ${fmt(c.sampled)} in this sample`}
          >
            <span className="box" />
            {mode === "country" && <span className="swatch" style={{ "--c": c.color } as React.CSSProperties} />}
            {pretty(c.name)}
            <span className="n">{fmt(perCountry[i])}</span>
          </button>
        ))}
      </section>

      {mode === "domain" && (
        <section>
          <h2>nearest of eight domains</h2>
          <ul className="legend">
            {data.domains.map((d, i) => (
              <li key={d.id}>
                <span className="swatch" style={{ "--c": d.color } as React.CSSProperties} />
                {d.label}
                <span className="n">{fmt(perDomain[i])}</span>
              </li>
            ))}
          </ul>
          <p className="foot">
            Domain = highest z-scored cosine similarity to eight domain{" "}
            {data.domainSource === "tei" ? "phrases embedded with the index model" : "keyword centroids"};
            muted points sit near a boundary.
          </p>
        </section>
      )}

      <p className="foot">
        {caption}
        {data.model ? ` Embeddings: ${data.model}.` : ""} Drag to orbit, scroll to zoom, right-drag to pan.
      </p>
    </aside>
  );
}
