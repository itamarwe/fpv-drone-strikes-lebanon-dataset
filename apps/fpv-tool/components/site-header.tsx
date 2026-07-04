import Link from "next/link";
import { withBasePath } from "@/lib/urls";

const navItems = [
  { key: "annotate", label: "Annotate", href: "/annotate/" },
  { key: "scenes", label: "Scenes", href: "/scenes/" },
] as const;

// Uniform top bar shared across every view (mirrors the scene viewer / browser).
export function SiteHeader({ active }: { active?: "annotate" | "scenes" }) {
  return (
    <header className="sticky top-0 z-20 flex h-[52px] items-center gap-2.5 border-b border-topbar-border bg-topbar px-3.5 backdrop-blur-md">
      <Link
        href={withBasePath("/")}
        className="whitespace-nowrap text-[15px] font-extrabold text-primary hover:opacity-80"
      >
        FPV Video
      </Link>
      <nav className="ml-auto flex gap-1">
        {navItems.map((item) => (
          <a
            key={item.key}
            href={withBasePath(item.href)}
            aria-current={active === item.key ? "page" : undefined}
            className={
              active === item.key
                ? "inline-flex min-h-[30px] items-center rounded-lg border border-primary bg-primary px-2.5 text-[13px] font-extrabold text-primary-foreground"
                : "inline-flex min-h-[30px] items-center rounded-lg border border-border px-2.5 text-[13px] text-muted-foreground transition-colors hover:text-foreground"
            }
          >
            {item.label}
          </a>
        ))}
      </nav>
    </header>
  );
}
