import time
from playwright.sync_api import sync_playwright
OUT = r"C:\Users\DELL\tmp_report_stage\shots3"
def wait_idle(page, extra=1.5, timeout=240):
    t0=time.time(); time.sleep(1.0)
    while time.time()-t0<timeout:
        if page.locator('[data-testid="stStatusWidget"]').count()==0: break
        time.sleep(0.5)
    time.sleep(extra)
def to_top(page, loc, offset=90):
    loc.scroll_into_view_if_needed(); time.sleep(0.4)
    loc.evaluate("(el,o)=>{let p=el; while(p && !(p.scrollHeight>p.clientHeight+5 && getComputedStyle(p).overflowY!='visible')) p=p.parentElement; const r=el.getBoundingClientRect(); if(p){p.scrollTop+= r.top-o;} else {window.scrollBy(0,r.top-o);} }", offset)
    time.sleep(1.2)
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
    for i in range(exps.count()):
        if "score=" in exps.nth(i).inner_text(): exps.nth(i).click(); time.sleep(1); break
    to_top(page, page.get_by_role("heading", name="Retrieved evidence"), 300)
    page.screenshot(path=OUT+r"\ui_02b_historical_evidence.png")
    page.locator('[data-testid="stSidebar"]').get_by_text("Comparative analysis").click(); wait_idle(page, 5)
    for name, fn in [("Significance","ui_04b_significance.png"),("Ablations (Adaptive only)","ui_04c_ablations.png")]:
        try:
            to_top(page, page.get_by_role("heading", name=name, exact=True).first, 80); page.screenshot(path=OUT+"\\"+fn)
        except Exception as e: print(name, e)
    charts = page.locator('[data-testid="stPlotlyChart"]')
    print("plotly", charts.count())
    if charts.count()>0:
        to_top(page, charts.first, 60); page.screenshot(path=OUT+r"\ui_04d_plotly_1.png")
        charts.first.screenshot(path=OUT+r"\ui_04d_plotly_1_el.png")
    if charts.count()>1:
        to_top(page, charts.nth(1), 60); page.screenshot(path=OUT+r"\ui_04e_plotly_2.png")
    b.close()
print("DONE")
