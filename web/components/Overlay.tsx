"use client";

/** The thin control panel: country toggles, domain legend, counts. Nothing else. */
import { useMemo } from "react";
import type { PointsPayload } from "@/lib/points";

const fmt = (n: number) => n.toLocaleString("en-US");

export default function Overlay({
  data,
  visible,
  enabled,
  onToggle,
  onAll,
}: {
  data: PointsPayload;
  visible: Uint8Array;
  enabled: ReadonlySet<string>;
  onToggle: (country: string) => void;
  onAll: (on: boolean) => void;
}) {
  const { shown, perDomain } = useMemo(() => {
    const perDomain = new Array<number>(data.domains.length).fill(0);
    let shown = 0;
    for (let i = 0; i < data.sample; i++) {
      if (!visible[i]) continue;
      shown++;
      perDomain[data.domain[i]]++;
    }
    return { shown, perDomain };
  }, [data, visible]);

  return (
    <aside className="panel" aria-label="controls">
      <h1>global-law-vector · embedding space</h1>
      <p className="sub">
        showing <span className="count">{fmt(shown)}</span> of {fmt(data.sample)} sampled points
        <br />
        from {fmt(data.totalChunks)} indexed chunks
      </p>

      <section>
        <h2>
          countries
          <span className="links">
            <button type="button" onClick={() => onAll(true)}>all</button>
            <button type="button" onClick={() => onAll(false)}>none</button>
          </span>
        </h2>
        {data.countries.map((c) => (
          <button
            key={c.name}
            type="button"
            className="toggle"
            aria-pressed={enabled.has(c.name)}
            onClick={() => onToggle(c.name)}
            title={`${fmt(c.total)} chunks indexed, ${fmt(c.sampled)} in this sample`}
          >
            <span className="box" />
            {c.name.replace(/-/g, " ")}
            <span className="n">{fmt(c.sampled)}</span>
          </button>
        ))}
      </section>

      <section>
        <h2>domain · nearest of eight</h2>
        <ul className="legend">
          {data.domains.map((d, i) => (
            <li key={d.id}>
              <span className="swatch" style={{ "--c": d.color } as React.CSSProperties} />
              {d.label}
              <span className="n">{fmt(perDomain[i])}</span>
            </li>
          ))}
        </ul>
      </section>

      <p className="foot">
        PCA of the 768-d Arctic embeddings, fitted on {fmt(data.projection.fitSize)} points. Colour is the
        closest domain {data.domainSource === "tei" ? "phrase, embedded with the same model" : "centroid, from keyword-matched chunks"};
        faint points are near a boundary. Drag to orbit, scroll to zoom, right-drag to pan.
      </p>
    </aside>
  );
}
