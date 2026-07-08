import type { NextConfig } from "next";

const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

const nextConfig: NextConfig = {
  basePath: basePath || undefined,
  trailingSlash: true,
  async rewrites() {
    const prefix = basePath || "";
    return [
      { source: `${prefix}/annotate`, destination: `${prefix}/annotate/index.html` },
      { source: `${prefix}/annotate/`, destination: `${prefix}/annotate/index.html` },
    ];
  },
};

export default nextConfig;
