/**
 * GET /api/points[?n=40000][&refresh=1]
 *
 * The visualisation sample as JSON (see `PointsPayload` in lib/points.ts): coordinates,
 * country, nearest domain and a confidence per point. Built on the first request and then
 * served from memory. `n` only affects the PCA fallback; a UMAP layout fixes its own size.
 * `refresh=1` rebuilds the cached payload.
 */
import { NextRequest, NextResponse } from "next/server";
import { env, POINTS_RANGE } from "@/lib/env";
import { getPoints } from "@/lib/points";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function GET(req: NextRequest) {
  try {
    const e = env();
    const asked = Number.parseInt(req.nextUrl.searchParams.get("n") ?? "", 10);
    const n = Number.isFinite(asked) ? Math.min(Math.max(asked, POINTS_RANGE.min), POINTS_RANGE.max) : e.points;
    const payload = await getPoints(n, req.nextUrl.searchParams.get("refresh") === "1");
    return NextResponse.json(payload, { headers: { "Cache-Control": "private, max-age=300" } });
  } catch (err) {
    // The full error goes to the server log. The client gets the message, which names the
    // failing step (connection refused, missing table, bad setting) but no connection string.
    const message = (err as Error).message;
    console.error("[glv-web] /api/points failed:", err);
    return NextResponse.json(
      { error: message, hint: "Check DATABASE_URL and GLV_SCHEMA, and that `glv index` has run. The server log has details." },
      { status: 500 },
    );
  }
}
