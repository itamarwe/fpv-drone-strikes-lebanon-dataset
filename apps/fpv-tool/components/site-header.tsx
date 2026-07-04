import Link from "next/link";
import { withBasePath } from "@/lib/config";

export function SiteHeader() {
  return (
    <header className="sticky top-0 z-20 border-b border-border bg-background/80 backdrop-blur-md">
      <div className="mx-auto flex max-w-[1600px] items-center gap-6 px-4 py-3 sm:px-6">
        <Link
          href={withBasePath("/")}
          className="text-[15px] font-semibold tracking-tight text-foreground transition-opacity hover:opacity-70"
        >
          FPV Video
        </Link>
        <nav className="ml-auto flex items-center gap-5 text-[14px] text-muted-foreground">
          <a href={withBasePath("/annotate/")} className="transition-colors hover:text-foreground">
            Annotate
          </a>
          <a href={withBasePath("/scenes/")} className="transition-colors hover:text-foreground">
            Scenes
          </a>
        </nav>
      </div>
    </header>
  );
}
