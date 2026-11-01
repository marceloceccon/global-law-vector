import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // `pg` is a Node module; keep it out of the client/edge bundles.
  serverExternalPackages: ["pg"],
  // Self-contained server in .next/standalone, which the Dockerfile copies alone.
  output: "standalone",
};

export default nextConfig;
