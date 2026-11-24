import time
from playwright.sync_api import sync_playwright
OUT=r"C:\Users\DELL\tmp_report_stage\shots4"
with sync_playwright() as p:
    b=p.chromium.launch(channel="msedge", headless=True)
    ctx=b.new_context(viewport={"width":1600,"height":1000}, device_scale_factor=1.5)
    pg=ctx.new_page()
    pg.goto("https://adaptive-temporal-rag.vercel.app", wait_until="networkidle"); time.sleep(2)
    q=pg.get_by_label("Question"); q.fill("Who is the current Prime Minister of India?")
    pg.get_by_role("button", name="Ask").click(); time.sleep(6)
    pg.screenshot(path=OUT+r"\ui_06_vercel_ask.png", full_page=True)
    pg.goto("https://adaptive-temporal-rag.vercel.app/results", wait_until="networkidle"); time.sleep(2)
    pg.screenshot(path=OUT+r"\ui_06b_vercel_results.png", full_page=True)
    b.close()
print("DONE")
