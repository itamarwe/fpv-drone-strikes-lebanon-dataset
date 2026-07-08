import { proxyToPython } from "@/lib/python-proxy";

type RouteContext = { params: Promise<{ path?: string[] }> };

function apiPath(params: { path?: string[] }): string {
  const segments = params.path ?? [];
  return `/api/${segments.join("/")}`;
}

export async function GET(request: Request, context: RouteContext) {
  const params = await context.params;
  return proxyToPython(request, apiPath(params));
}

export async function POST(request: Request, context: RouteContext) {
  const params = await context.params;
  return proxyToPython(request, apiPath(params));
}

export async function PUT(request: Request, context: RouteContext) {
  const params = await context.params;
  return proxyToPython(request, apiPath(params));
}

export async function DELETE(request: Request, context: RouteContext) {
  const params = await context.params;
  return proxyToPython(request, apiPath(params));
}
