"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

type Theme = "light" | "dark";

export function SiteHeader() {
  const path = usePathname();
  const [theme, setTheme] = useState<Theme | null>(null);

  useEffect(() => {
    setTheme(document.documentElement.dataset.theme === "dark" ? "dark" : "light");
  }, []);

  function choose(next: Theme) {
    setTheme(next);
    document.documentElement.dataset.theme = next;
    localStorage.setItem("theme", next);
  }

  return (
    <header className="top">
      <Link className="brand" href="/">Adaptive Temporal RAG</Link>
      <div className="top-actions">
        <nav className="nav">
          <Link href="/" data-active={path === "/" ? "true" : "false"}>Ask</Link>
          <Link href="/results" data-active={path === "/results" ? "true" : "false"}>Results</Link>
        </nav>
        <div className="theme" role="group" aria-label="Color mode">
          <button type="button" aria-pressed={theme === "light"} onClick={() => choose("light")}>Light</button>
          <button type="button" aria-pressed={theme === "dark"} onClick={() => choose("dark")}>Dark</button>
        </div>
      </div>
    </header>
  );
}
