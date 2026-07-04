import Link from "next/link";
import { withBasePath } from "@/lib/config";

export function SiteHeader() {
  return (
    <header className="sticky top-0 z-20 border-b border-border bg-background/85 backdrop-blur">
      <div className="mx-auto flex max-w-7xl items-center gap-4 px-4 py-3">
        <Link
          href={withBasePath("/")}
          className="text-lg font-semibold tracking-tight text-primary hover:opacity-80"
        >
          FPV Video
        </Link>
        <nav className="ml-auto flex items-center gap-1 text-sm">
          <a
            href={withBasePath("/annotate/")}
            className="rounded-md px-3 py-1.5 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
          >
            Annotate
          </a>
          <a
            href={withBasePath("/scenes/")}
            className="rounded-md px-3 py-1.5 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
          >
            Scenes
          </a>
        </nav>
      </div>
    </header>
  );
}
