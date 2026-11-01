import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // `pg` is a Node module; keep it out of the client/edge bundles.
  serverExternalPackages: ["pg"],
};

export default nextConfig;
