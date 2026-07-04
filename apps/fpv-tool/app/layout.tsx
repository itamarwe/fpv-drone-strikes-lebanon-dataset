import type { ReactNode } from "react";
import "./globals.css";

export const metadata = {
  title: "FPV Video",
  description: "Annotate FPV strike videos and inspect reconstructed 3D scenes.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
