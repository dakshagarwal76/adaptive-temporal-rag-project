"""Draw clean academic architecture diagrams for the mid-term report."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

NAVY = (26, 54, 93)
TEAL = (43, 108, 176)
LIGHT = (236, 244, 252)
WHITE = (255, 255, 255)
GRAY = (74, 85, 104)
BLACK = (26, 26, 26)
GREEN = (39, 103, 73)
ORANGE = (154, 82, 14)


def font(size, bold=False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def rounded_box(draw, xy, fill, outline, width=2, radius=14):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def center_text(draw, cx, cy, text, fnt, fill=BLACK):
    bbox = draw.textbbox((0, 0), text, font=fnt)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text((cx - w / 2, cy - h / 2), text, font=fnt, fill=fill)


def multiline_center(draw, cx, cy, lines, fnt, fill=BLACK, gap=4):
    heights = []
    widths = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=fnt)
        widths.append(bbox[2] - bbox[0])
        heights.append(bbox[3] - bbox[1])
    total_h = sum(heights) + gap * (len(lines) - 1)
    y = cy - total_h / 2
    for line, w, h in zip(lines, widths, heights):
        draw.text((cx - w / 2, y), line, font=fnt, fill=fill)
        y += h + gap


def arrow(draw, x1, y1, x2, y2, color=NAVY):
    draw.line((x1, y1, x2, y2), fill=color, width=3)
    # arrow head
    if x2 >= x1:
        draw.polygon([(x2, y2), (x2 - 10, y2 - 6), (x2 - 10, y2 + 6)], fill=color)
    else:
        draw.polygon([(x2, y2), (x2 + 10, y2 - 6), (x2 + 10, y2 + 6)], fill=color)


def draw_temporal():
    W, H = 1600, 900
    img = Image.new("RGB", (W, H), WHITE)
    d = ImageDraw.Draw(img)
    title = font(36, True)
    box_f = font(18, True)
    small = font(15)
    note = font(16, True)

    center_text(d, W // 2, 42, "Figure: Temporal RAG Architecture (Fixed Baseline)", title, NAVY)
    center_text(
        d,
        W // 2,
        78,
        "Always retrieves  |  Fixed temporal rules  |  No query analyzer",
        small,
        GRAY,
    )

    boxes = [
        (40, 360, 220, 520, ["User Query"], LIGHT),
        (260, 360, 470, 520, ["Dense Embedder", "(MiniLM)"], LIGHT),
        (510, 360, 760, 520, ["FAISS Retrieval", "Always top-k", "candidates"], LIGHT),
        (800, 220, 1120, 400, ["Year present?", "Historical window", "(+/- 2 years)"], (255, 246, 230)),
        (800, 460, 1120, 640, ["No year?", "Recency boost", "(prefer newer docs)"], (230, 247, 240)),
        (1160, 360, 1380, 520, ["Score Fusion", "a*sem + (1-a)*temp", "a = 0.6"], LIGHT),
        (1420, 280, 1560, 400, ["Top-k", "Evidence"], LIGHT),
        (1420, 460, 1560, 620, ["Generator", "Answer"], LIGHT),
    ]
    for x1, y1, x2, y2, lines, fill in boxes:
        rounded_box(d, (x1, y1, x2, y2), fill, NAVY, 3)
        multiline_center(d, (x1 + x2) / 2, (y1 + y2) / 2, lines, box_f, NAVY)

    # arrows
    arrow(d, 220, 440, 260, 440)
    arrow(d, 470, 440, 510, 440)
    arrow(d, 760, 420, 800, 310)
    arrow(d, 760, 460, 800, 550)
    arrow(d, 1120, 310, 1220, 400)
    arrow(d, 1120, 550, 1220, 480)
    arrow(d, 1380, 420, 1420, 340)
    arrow(d, 1490, 400, 1490, 460)

    # policy label
    rounded_box(d, (800, 160, 1120, 205), NAVY, NAVY, 1, 10)
    center_text(d, 960, 182, "Temporal Policy Module", note, WHITE)

    footer = (
        "Pipeline: Query -> Embed -> Always Retrieve -> Fixed Temporal Policy -> Fuse Scores -> Generate"
    )
    center_text(d, W // 2, H - 40, footer, small, GRAY)
    path = OUT / "architecture_temporal_rag.jpg"
    img.save(path, quality=95)
    return path


def draw_adaptive():
    W, H = 1600, 980
    img = Image.new("RGB", (W, H), WHITE)
    d = ImageDraw.Draw(img)
    title = font(34, True)
    box_f = font(17, True)
    small = font(14)
    tiny = font(13)

    center_text(d, W // 2, 36, "Figure: Adaptive Temporal RAG Architecture (Proposed)", title, NAVY)
    center_text(
        d,
        W // 2,
        70,
        "Conditional retrieval  |  Class-specific temporal policy  |  Confidence-aware fallback",
        small,
        GRAY,
    )

    # Query
    rounded_box(d, (40, 140, 200, 260), LIGHT, NAVY, 3)
    multiline_center(d, 120, 200, ["User", "Query"], box_f, NAVY)

    # Analyzer
    rounded_box(d, (240, 110, 560, 290), (232, 240, 254), NAVY, 3)
    multiline_center(
        d,
        400,
        160,
        ["DistilBERT Multitask", "Query Analyzer"],
        box_f,
        NAVY,
    )
    # three heads
    heads = [
        (260, 210, 350, 270, ["Necessity"]),
        (360, 210, 450, 270, ["Temporal", "Class"]),
        (460, 210, 540, 270, ["Confidence"]),
    ]
    for x1, y1, x2, y2, lines in heads:
        rounded_box(d, (x1, y1, x2, y2), WHITE, TEAL, 2, 8)
        multiline_center(d, (x1 + x2) / 2, (y1 + y2) / 2, lines, tiny, TEAL)

    arrow(d, 200, 200, 240, 200)

    # Policy diamond-ish box
    rounded_box(d, (600, 130, 860, 270), (255, 248, 230), ORANGE, 3)
    multiline_center(d, 730, 200, ["Policy Decision", "Module"], box_f, ORANGE)
    arrow(d, 560, 200, 600, 200)

    # Branches
    branches = [
        (40, 360, 360, 470, ["needs_retrieval = 0", "NO RETRIEVAL", "Parametric generate"], (237, 247, 237)),
        (400, 360, 720, 470, ["confidence < 0.55", "OR timeless", "Semantic-only rank"], (236, 244, 252)),
        (760, 360, 1080, 470, ["class = current", "Recency re-rank", "prefer newer docs"], (255, 246, 230)),
        (1120, 360, 1560, 470, ["class = historical", "Year window (+/-2)", "then temporal re-rank"], (255, 240, 240)),
    ]
    for x1, y1, x2, y2, lines, fill in branches:
        rounded_box(d, (x1, y1, x2, y2), fill, NAVY, 2)
        multiline_center(d, (x1 + x2) / 2, (y1 + y2) / 2, lines, small, NAVY)

    # lines from policy to branches
    for cx in [200, 560, 920, 1340]:
        d.line((730, 270, 730, 320), fill=NAVY, width=2)
        d.line((730, 320, cx, 320), fill=NAVY, width=2)
        arrow(d, cx, 320, cx, 360)

    # Shared retrieve stack
    rounded_box(d, (200, 540, 1400, 700), LIGHT, NAVY, 3)
    center_text(d, 800, 565, "When retrieval is selected", font(16, True), NAVY)
    steps = [
        (230, 590, 430, 680, ["MiniLM", "Embed"]),
        (470, 590, 700, 680, ["FAISS", "Candidates ~50"]),
        (740, 590, 1020, 680, ["Temporal / Semantic", "Re-ranker"]),
        (1060, 590, 1260, 680, ["Top-k=5", "Evidence"]),
        (1300, 590, 1370, 680, ["Gen"]),
    ]
    for i, (x1, y1, x2, y2, lines) in enumerate(steps):
        rounded_box(d, (x1, y1, x2, y2), WHITE, TEAL, 2, 10)
        multiline_center(d, (x1 + x2) / 2, (y1 + y2) / 2, lines, small, TEAL)
        if i < len(steps) - 1:
            arrow(d, x2, (y1 + y2) / 2, steps[i + 1][0], (y1 + y2) / 2, TEAL)

    # arrows from retrieve branches into stack
    for cx in [560, 920, 1340]:
        d.line((cx, 470, cx, 510), fill=GRAY, width=2)
        d.line((cx, 510, 800, 510), fill=GRAY, width=2)
    arrow(d, 800, 510, 800, 540, GRAY)

    # no-retrieval path to answer
    rounded_box(d, (40, 760, 360, 880), (237, 247, 237), GREEN, 3)
    multiline_center(d, 200, 820, ["Answer", "(no documents)"], box_f, GREEN)
    arrow(d, 200, 470, 200, 760, GREEN)

    rounded_box(d, (1120, 760, 1560, 880), (237, 247, 237), GREEN, 3)
    multiline_center(d, 1340, 820, ["Answer", "(with dated evidence)"], box_f, GREEN)
    arrow(d, 1335, 700, 1335, 760, GREEN)

    center_text(
        d,
        W // 2,
        H - 36,
        "final_score = a * semantic_score + (1 - a) * temporal_score   |   default a = 0.6   |   conf_threshold = 0.55",
        small,
        GRAY,
    )

    path = OUT / "architecture_adaptive_temporal_rag.jpg"
    img.save(path, quality=95)
    return path


if __name__ == "__main__":
    p1 = draw_temporal()
    p2 = draw_adaptive()
    print(p1)
    print(p2)
