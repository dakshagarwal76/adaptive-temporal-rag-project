"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

export function SiteHeader() {
  const path = usePathname();
  return (
    <header className="top">
      <Link className="brand" href="/">Adaptive Temporal RAG</Link>
      <nav className="nav">
        <Link href="/" data-active={path === "/" ? "true" : "false"}>Ask</Link>
        <Link href="/results" data-active={path === "/results" ? "true" : "false"}>Results</Link>
      </nav>
    </header>
  );
}
