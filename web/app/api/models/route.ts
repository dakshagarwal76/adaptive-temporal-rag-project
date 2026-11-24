import { NextResponse } from "next/server";
import { configuredModels } from "@/lib/generate";

export async function GET() {
  return NextResponse.json({ models: configuredModels() });
}
