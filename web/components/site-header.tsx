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
        <button
          type="button"
          className="theme-toggle"
          role="switch"
          aria-checked={theme === "dark"}
          aria-label="Color mode"
          onClick={() => choose(theme === "dark" ? "light" : "dark")}
        >
          <span className="theme-track" aria-hidden="true"><span className="theme-thumb" /></span>
          {theme === "dark" ? "Dark" : "Light"}
        </button>
      </div>
    </header>
  );
}
