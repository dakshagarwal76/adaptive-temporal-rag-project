from pathlib import Path
from docx import Document

p = Path(r"c:\Users\Assessment\Documents\cursor\DOC\ARP455_Progress_Report_Adaptive_Temporal_RAG.docx")
doc = Document(p)
repls = {
    "\u2013": "-",
    "\u2014": "-",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2026": "...",
    "\u00a0": " ",
}
count = 0


def fix_runs(container):
    global count
    for para in container.paragraphs:
        for run in para.runs:
            t = run.text
            nt = t
            for a, b in repls.items():
                if a in nt:
                    nt = nt.replace(a, b)
            if nt != t:
                run.text = nt
                count += 1


fix_runs(doc)
for table in doc.tables:
    for row in table.rows:
        for cell in row.cells:
            fix_runs(cell)

doc.save(p)
print("fixed runs", count)
for para in doc.paragraphs:
    if "Mid-Term" in para.text:
        print(para.text)
