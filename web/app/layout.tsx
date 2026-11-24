import type { Metadata } from "next";
import "./globals.css";
import { SiteHeader } from "@/components/site-header";

export const metadata: Metadata = {
  title: "Adaptive Temporal RAG",
  description: "Ask a time-sensitive question. The system decides whether to retrieve and which dates to keep.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="shell">
          <SiteHeader />
          {children}
          <p className="foot">USAR minor project. Answers come from the project’s synthetic dated corpus.</p>
        </div>
      </body>
    </html>
  );
}
