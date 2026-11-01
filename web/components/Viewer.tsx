"use client";

/**
 * Client shell: fetches the sample once, owns the country toggle state and hands both to the
 * canvas and the overlay. The canvas is loaded client-side only (WebGL has no server render).
 */
import dynamic from "next/dynamic";
import { useCallback, useEffect, useMemo, useState } from "react";
import type { PointsPayload } from "@/lib/points";
import Overlay from "./Overlay";

const PointCloud = dynamic(() => import("./PointCloud"), { ssr: false });

type Status = { kind: "loading" } | { kind: "ready" } | { kind: "error"; message: string; hint?: string };

export default function Viewer() {
  const [data, setData] = useState<PointsPayload | null>(null);
  const [status, setStatus] = useState<Status>({ kind: "loading" });
  const [enabled, setEnabled] = useState<ReadonlySet<string>>(new Set());

  useEffect(() => {
    const ctl = new AbortController();
    (async () => {
      try {
        const res = await fetch("/api/points", { signal: ctl.signal });
        const body = await res.json();
        if (!res.ok) throw Object.assign(new Error(body.error ?? res.statusText), { hint: body.hint });
        const payload = body as PointsPayload;
        setData(payload);
        setEnabled(new Set(payload.countries.map((c) => c.name)));
        setStatus({ kind: "ready" });
      } catch (err) {
        if (ctl.signal.aborted) return;
        setStatus({ kind: "error", message: (err as Error).message, hint: (err as { hint?: string }).hint });
      }
    })();
    return () => ctl.abort();
  }, []);

  const toggle = useCallback((name: string) => {
    setEnabled((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  }, []);
  const setAll = useCallback(
    (on: boolean) => setEnabled(new Set(on && data ? data.countries.map((c) => c.name) : [])),
    [data],
  );

  // Per-point visibility, recomputed only when a toggle changes: the geometry stays put.
  const visible = useMemo(() => {
    if (!data) return null;
    const on = data.countries.map((c) => enabled.has(c.name));
    const mask = new Uint8Array(data.sample);
    for (let i = 0; i < data.sample; i++) mask[i] = on[data.country[i]] ? 1 : 0;
    return mask;
  }, [data, enabled]);

  return (
    <>
      <div className="stage">{data && visible && <PointCloud data={data} visible={visible} />}</div>
      {status.kind === "loading" && (
        <div className="notice">
          <div className="pulse">sampling the index and fitting the projection…</div>
        </div>
      )}
      {status.kind === "error" && (
        <div className="notice">
          <div className="err">
            <div>Could not load the point sample.</div>
            <p><code>{status.message}</code></p>
            {status.hint && <div style={{ color: "var(--muted)" }}>{status.hint}</div>}
          </div>
        </div>
      )}
      {data && visible && <Overlay data={data} visible={visible} enabled={enabled} onToggle={toggle} onAll={setAll} />}
    </>
  );
}
