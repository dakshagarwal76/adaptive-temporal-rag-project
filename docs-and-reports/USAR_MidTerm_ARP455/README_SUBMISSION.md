# USAR Mid-Term Submission Pack (ARP-455)

Prepared for **1st Mid-Term Evaluation of Minor Project (ARP-455)**  
Guru Gobind Singh Indraprastha University — University School of Automation & Robotics (USAR)  
Evaluation date: **8 October 2026**

## What to submit / carry

| File | Purpose |
|------|---------|
| `ARP455_Progress_Report_Adaptive_Temporal_RAG.docx` | **Progress Report** (all 11 mandatory sections) — get supervisor signature on the cover page |
| `ARP455_MidTerm_Presentation_Adaptive_Temporal_RAG.pptx` | **PowerPoint** (10 slides) for the evaluation |
| `figures/` | Architecture + results images (already embedded in DOCX/PPTX) |

## Notice checklist (from USAR notice dated 5 Oct 2026)

Progress Report includes:

1. Title of the Project  
2. Introduction  
3. Literature Survey  
4. Problem Statement  
5. Motivation  
6. Proposed Design Flow — **Temporal RAG** + **Adaptive Temporal RAG** architectures with diagrams  
7. Proposed Methodology  
8. Results (if any)  
9. GANTT / PERT Chart  
10. Hardware and Software Requirements  
11. References  

Also includes **Supervisor Certification** block for signature.

## Student details used

- **Authors:** Daksh Agarwal (10819051623) and Shashank Kumar (08619051623)  
- **Programme:** B.Tech. (AI & ML), Batch B2  
- **Guide:** Dr. Annu Priya, Assistant Professor, USAR  

If any detail needs correction, edit the Word/PowerPoint files or re-run:

```powershell
cd D:\cursor\USAR_MidTerm_ARP455
python make_architecture_diagrams.py
python generate_documents.py
```

## Tip for viva

Be ready to contrast:

- **Temporal RAG (B3):** always retrieves; fixed year/recency rules  
- **Adaptive Temporal RAG:** DistilBERT decides whether to retrieve and which temporal policy to use  
