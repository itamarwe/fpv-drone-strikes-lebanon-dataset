import Link from "next/link";
import { withBasePath } from "@/lib/config";

export default function HomePage() {
  return (
    <main>
      <h1>FPV Scene Tool</h1>
      <p>
        Next.js front door for the FPV annotation and 3D scene viewer. Heavy VGGT
        reconstruction still runs in the Python backend.
      </p>
      <nav>
        <Link href={withBasePath("/annotate/")}>Annotate</Link>
        <Link href={withBasePath("/scenes/")}>Scenes</Link>
      </nav>
    </main>
  );
}
