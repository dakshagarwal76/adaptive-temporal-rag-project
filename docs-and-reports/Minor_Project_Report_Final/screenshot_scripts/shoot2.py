import time
from playwright.sync_api import sync_playwright
OUT = r"C:\Users\DELL\tmp_report_stage\shots2"
def wait_idle(page, extra=1.5, timeout=240):
    t0=time.time(); time.sleep(1.0)
    while time.time()-t0<timeout:
        if page.locator('[data-testid="stStatusWidget"]').count()==0: break
        time.sleep(0.5)
    time.sleep(extra)
def wheel(page, dy):
    page.mouse.move(1000, 600); page.mouse.wheel(0, dy); time.sleep(1.2)
with sync_playwright() as p:
    b = p.chromium.launch(channel="msedge", headless=True)
    ctx = b.new_context(viewport={"width":1600,"height":1000}, device_scale_factor=1.5)
    page = ctx.new_page()
    page.goto("http://localhost:8505", wait_until="networkidle")
    page.get_by_text("Index ready").wait_for(timeout=300000); wait_idle(page, 2)
    box = page.get_by_label("Ask a time-sensitive question")
    box.fill("Who was the President of the United States in 2018?"); box.press("Enter"); time.sleep(0.8)
    page.get_by_role("button", name="Run").click(); wait_idle(page, 2.5)
    exps = page.locator('[data-testid="stExpander"] summary')
    n = exps.count(); print("expanders", n)
    for i in range(n):
        t = exps.nth(i).inner_text()
        if "score=" in t:
            exps.nth(i).click(); time.sleep(1); break
    page.get_by_text("Retrieved evidence").scroll_into_view_if_needed(); time.sleep(0.5)
    wheel(page, -150)
    page.screenshot(path=OUT+r"\ui_02b_historical_evidence.png")
    page.locator('[data-testid="stSidebar"]').get_by_text("Comparative analysis").click(); wait_idle(page, 5)
    for i,dy in enumerate([1700,1700,1700,1700,1700,1700]):
        wheel(page, dy); page.screenshot(path=OUT+rf"\ui_04_scroll_{i+1}.png")
    # web UI
    page2 = ctx.new_page()
    page2.goto("http://localhost:3005", wait_until="networkidle"); time.sleep(2)
    page2.screenshot(path=OUT+r"\ui_05_web_home.png")
    q = page2.get_by_label("Question"); q.fill("Who was the President of the United States in 2018?")
    page2.get_by_role("button", name="Ask").click(); time.sleep(6)
    page2.screenshot(path=OUT+r"\ui_05_web_answer.png", full_page=True)
    page2.goto("http://localhost:3005/results", wait_until="networkidle"); time.sleep(2)
    page2.screenshot(path=OUT+r"\ui_05_web_results.png", full_page=True)
    b.close()
print("DONE")
