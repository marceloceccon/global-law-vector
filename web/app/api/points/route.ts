/**
 * GET /api/points?n=12000[&refresh=1]
 * The visualisation sample: projected coordinates, country, corpus, nearest domain and a
 * confidence per point. Served from an in-process cache after the first build (a few seconds).
 */
import { NextRequest, NextResponse } from "next/server";
import { env } from "@/lib/env";
import { getPoints } from "@/lib/points";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function GET(req: NextRequest) {
  const e = env();
  const n = Math.min(Math.max(Number(req.nextUrl.searchParams.get("n")) || e.points, 500), 60000);
  const refresh = req.nextUrl.searchParams.get("refresh") === "1";
  try {
    const payload = await getPoints(n, refresh);
    return NextResponse.json(payload, { headers: { "Cache-Control": "private, max-age=300" } });
  } catch (err) {
    const message = (err as Error).message;
    console.error("[glv-web] /api/points failed:", message);
    return NextResponse.json(
      { error: message, hint: `database ${e.databaseUrl.replace(/\/\/.*@/, "//…@")} schema ${e.schema}` },
      { status: 500 },
    );
  }
}
