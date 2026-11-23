# Architecture diagrams (PNG)

| File | Description |
|------|-------------|
| `01_adaptive_inference_full.png` | Adaptive system inference end-to-end |
| `02_policy_router.png` | Analyzer → retrieve mode policy |
| `03_colab_four_systems_eval.png` | Colab section 4: B1–B4 on same test queries |

**In-depth explanation of every node:** [INFERENCE_NODES.md](INFERENCE_NODES.md)

**Sources:** matching `.mmd` files (Mermaid).

**Regenerate PNGs** (requires network):

```powershell
.\scripts\render_diagram_pngs.ps1
```

Uses [Kroki](https://kroki.io) to render Mermaid.
