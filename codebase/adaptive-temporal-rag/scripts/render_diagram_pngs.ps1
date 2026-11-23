# Render docs/diagrams/*.mmd to PNG via Kroki (https://kroki.io)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$diagramDir = Join-Path $root "docs\diagrams"
if (-not (Test-Path $diagramDir)) {
    throw "Missing $diagramDir"
}

$mmdFiles = Get-ChildItem -Path $diagramDir -Filter "*.mmd" | Sort-Object Name
foreach ($mmd in $mmdFiles) {
    $png = Join-Path $diagramDir ($mmd.BaseName + ".png")
    $body = Get-Content -Path $mmd.FullName -Raw -Encoding UTF8
    Write-Host "Rendering $($mmd.Name) -> $($mmd.BaseName).png"
    Invoke-WebRequest -Uri "https://kroki.io/mermaid/png" -Method POST -Body $body -ContentType "text/plain; charset=utf-8" -OutFile $png -UseBasicParsing
}
Write-Host "Done. PNGs in $diagramDir"
