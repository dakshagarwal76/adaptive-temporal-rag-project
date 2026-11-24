import sys, time
from playwright.sync_api import sync_playwright
OUT = r"C:\Users\DELL\tmp_report_stage\shots"
URL = "http://localhost:8505"

def wait_idle(page, extra=1.5, timeout=240):
    t0 = time.time()
    time.sleep(1.0)
    while time.time() - t0 < timeout:
        running = page.locator('[data-testid="stStatusWidget"]').count()
        if running == 0:
            break
        time.sleep(0.5)
    time.sleep(extra)

def scroll_main(page, y):
    page.evaluate("""(y)=>{const m=document.querySelector('section[data-testid="stMain"]')||document.querySelector('[data-testid="stAppViewContainer"] section.main')||document.querySelector('section.main'); if(m){m.scrollTo(0,y);} window.scrollTo(0,y);}""", y)
    time.sleep(0.8)

def ask(page, q, system=None):
    if system:
        sb = page.locator('[data-testid="stSidebar"]')
        sel = sb.locator('[data-testid="stSelectbox"]').nth(1)
        sel.click(); time.sleep(0.6)
        page.get_by_role("option", name=system, exact=True).click(); wait_idle(page)
    box = page.get_by_label("Ask a time-sensitive question")
    box.fill(q); box.press("Enter"); time.sleep(0.8)
    page.get_by_role("button", name="Run").click()
    wait_idle(page, extra=2.5)

with sync_playwright() as p:
    b = p.chromium.launch(channel="msedge", headless=True)
    ctx = b.new_context(viewport={"width": 1600, "height": 1000}, device_scale_factor=1.5)
    page = ctx.new_page()
    page.goto(URL, wait_until="networkidle")
    page.get_by_text("Index ready").wait_for(timeout=300000)
    wait_idle(page, extra=2)
    page.screenshot(path=OUT + r"\ui_01_home_overview.png")
    ask(page, "Who was the President of the United States in 2018?")
    page.screenshot(path=OUT + r"\ui_02_historical_query.png")
    try:
        page.locator('[data-testid="stExpander"] summary').first.click(); time.sleep(1)
    except Exception as e: print("exp", e)
    scroll_main(page, 2000)
    page.screenshot(path=OUT + r"\ui_02b_historical_evidence.png")
    scroll_main(page, 0)
    ask(page, "Who is the current Prime Minister of India?")
    page.screenshot(path=OUT + r"\ui_03_current_query.png")
    ask(page, "What is 2 + 2?")
    page.screenshot(path=OUT + r"\ui_03b_timeless_query.png")
    page.locator('[data-testid="stSidebar"]').get_by_text("Comparative analysis").click()
    wait_idle(page, extra=4)
    page.screenshot(path=OUT + r"\ui_04_comparison_benchmark.png")
    scroll_main(page, 2600)
    page.screenshot(path=OUT + r"\ui_04b_comparison_charts.png")
    scroll_main(page, 5200)
    page.screenshot(path=OUT + r"\ui_04c_comparison_charts.png")
    b.close()
print("DONE")
