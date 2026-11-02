/**
 * GET /api/points
 *
 * The visualisation sample as JSON (see `PointsPayload` in lib/points.ts): coordinates,
 * country, nearest domain and a confidence per point. Built on the first request and then
 * served from memory; see `getPoints` for when it is rebuilt. The route takes no parameters,
 * so a visitor cannot make the server do more work than serving that one cached payload.
 */
import { randomUUID } from "node:crypto";
import { NextResponse } from "next/server";
import { env } from "@/lib/env";
import { getPoints } from "@/lib/points";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function GET() {
  try {
    const payload = await getPoints(env().points);
    return NextResponse.json(payload, { headers: { "Cache-Control": "private, max-age=300" } });
  } catch (err) {
    // Details stay in the server log. A production build tells the client only that it
    // failed and which log line to look for; `next dev` also shows the message.
    const ref = randomUUID().slice(0, 8);
    console.error(`[glv-web] /api/points failed (ref ${ref}):`, err);
    const detail = process.env.NODE_ENV === "production" ? null : (err as Error).message;
    return NextResponse.json(
      {
        error: detail ?? `The sample could not be built (ref ${ref}).`,
        hint: "The server log has the details. Check DATABASE_URL and GLV_SCHEMA, and that `glv index` has run.",
      },
      { status: 500 },
    );
  }
}
