#!/usr/bin/env python3
"""Drive the UI end to end in the installed Google Chrome: live uploads, every view, screenshots, frame rate.

  PYTHONPATH=src python scripts/ui_screenshots.py --geotiff <NAIP GeoTIFF> --png <image.png> \
      [--reference <reference DSM GeoTIFF>] [--headless]

Needs the server on 127.0.0.1:8000 (scripts/run_app.sh). Nothing is pre-computed: both jobs are uploaded through
the form, so the Processing screenshots show the real server event stream. Frame rate is the viewer's own counter
(window.__dw.fps) while orbiting with the mouse; headless Chrome may fall back to software rendering.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8000/app/"


def orbit_fps(page, seconds: float = 4.0) -> float:
    box = page.locator("#viewport").bounding_box()
    cx, cy = box["x"] + box["width"] * 0.55, box["y"] + box["height"] * 0.5
    page.mouse.move(cx, cy)
    page.mouse.down()
    t0, samples = time.time(), []
    while time.time() - t0 < seconds:
        dx = 200 * ((time.time() - t0) % 2 - 1)
        page.mouse.move(cx + dx, cy + dx * 0.2, steps=4)
        samples.append(page.evaluate("window.__dw.fps"))
    page.mouse.up()
    good = [s for s in samples if s]
    return float(sorted(good)[len(good) // 2]) if good else 0.0


STAGES = """() => [...document.querySelectorAll('.st')].map(li => ({stage: li.dataset.stage,
  status: li.className.replace('st', '').trim() || 'pending', secs: li.querySelector('.secs').textContent,
  summary: li.querySelector('.sum').textContent, thumbs: li.querySelectorAll('.thumbs img').length}))"""


def click_at(page, sel: str, fx: float, fy: float, wait: int = 800) -> None:
    box = page.locator(sel).bounding_box()
    page.mouse.click(box["x"] + box["width"] * fx, box["y"] + box["height"] * fy)
    page.wait_for_timeout(wait)


def upload(page, path: str, band_order: str = "auto") -> None:
    page.set_input_files("#file", path)
    page.select_option("#band_order", band_order)
    page.click("#go")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--geotiff", required=True)
    ap.add_argument("--png", required=True)
    ap.add_argument("--reference")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "docs" / "screenshots"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep: dict = {"headless": a.headless, "shots": [], "timings_s": {}}

    def shot(page, name, **kw):
        page.evaluate("() => getSelection().removeAllRanges()")
        page.screenshot(path=out / name, **kw)
        rep["shots"].append(name)

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=a.headless,
                                    args=["--ignore-gpu-blocklist", "--enable-gpu-rasterization"])
        page = browser.new_page(viewport={"width": 1680, "height": 980})
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: m.type == "error" and errors.append(m.text))
        requests: list[str] = []
        page.on("request", lambda r: requests.append(r.url))
        page.goto(BASE)
        page.wait_for_timeout(800)
        rep["webgl_renderer"] = page.evaluate("""() => { const g = document.createElement('canvas').getContext('webgl2');
            const e = g && g.getExtension('WEBGL_debug_renderer_info'); return e ? g.getParameter(e.UNMASKED_RENDERER_WEBGL) : 'unknown'; }""")

        # ---- 1. live GeoTIFF job: processing view driven by the server's event stream
        t0 = time.time()
        upload(page, a.geotiff)
        page.wait_for_selector("#view-process:not([hidden])", timeout=30000)
        page.wait_for_selector('.st[data-stage="calibration"].running, .st[data-stage="calibration"].done', timeout=300000)
        shot(page, "01_processing_live.png")
        rep["stages_mid_run"] = page.evaluate(STAGES)
        page.wait_for_selector("#open-3d:not([hidden])", timeout=600000)
        rep["timings_s"]["geotiff_job_wall"] = round(time.time() - t0, 1)
        page.wait_for_timeout(1500)                       # last thumbnails load
        shot(page, "02_processing_complete.png", full_page=False)
        rep["stages_final"] = page.evaluate(STAGES)
        rep["proc_clock"] = page.locator("#proc-clock").inner_text()

        # ---- 2. 3D flythrough
        page.click("#open-3d")
        page.wait_for_function("window.__dw.state.terrain !== null", timeout=60000)
        page.wait_for_timeout(2600)                       # fly-in
        shot(page, "03_3d_metric_overview.png")
        rep["fps_orbit_metric"] = orbit_fps(page)
        page.wait_for_timeout(400)
        click_at(page, "#viewport", 0.52, 0.58, 1200)
        shot(page, "04_3d_inspect_confidence.png")
        rep["inspect_readout"] = page.locator("#fb-dl").inner_text()
        rep["confidence_badge"] = page.locator("#fb-conf").inner_text() if page.locator("#fb-conf").is_visible() else None
        page.click('[data-tool="measure"]')
        click_at(page, "#viewport", 0.42, 0.47)
        click_at(page, "#viewport", 0.64, 0.62)
        shot(page, "05_3d_measure.png")
        rep["measure_readout"] = page.locator("#fb-dl").inner_text()
        page.click('[data-tool="inspect"]')
        page.fill("#exag", "3"); page.dispatch_event("#exag", "input")
        page.click('[data-overlay="slope"]')
        page.wait_for_timeout(900)
        shot(page, "06_3d_slope_legend.png")
        page.fill("#exag", "1"); page.dispatch_event("#exag", "input")
        if a.reference:
            page.set_input_files("#ref-file", a.reference)
            page.wait_for_function("!document.querySelector('[data-overlay=\"diff\"]').disabled", timeout=180000)
            page.wait_for_timeout(1500)
            shot(page, "07_3d_reference_difference.png")
            rep["reference_readout"] = page.locator("#ref-dl").inner_text()
        page.click('[data-overlay="photo"]')

        # ---- 3. layers: swipe compare + filmstrip + pixel inspector
        page.keyboard.press("2")
        page.wait_for_timeout(900)
        page.evaluate("() => { window.__dw.c2.split = 0.5; }")
        page.evaluate("() => document.querySelector('.film[data-id=\"rgb\"] .ab button').click()")
        page.evaluate("() => document.querySelectorAll('.film[data-id=\"dsm\"] .ab button')[1].click()")
        page.wait_for_timeout(900)
        shot(page, "08_layers_swipe_rgb_dsm.png")
        box = page.locator("#divider").bounding_box()
        page.mouse.move(box["x"] + 1, box["y"] + box["height"] / 2)
        page.mouse.down(); page.mouse.move(box["x"] - 260, box["y"] + box["height"] / 2, steps=8); page.mouse.up()
        page.evaluate("() => document.querySelectorAll('.film[data-id=\"ndsm\"] .ab button')[1].click()")
        cmp = page.locator("#compare").bounding_box()
        page.mouse.move(cmp["x"] + cmp["width"] * 0.55, cmp["y"] + cmp["height"] * 0.45)
        for _ in range(4):
            page.mouse.wheel(0, -240); page.wait_for_timeout(80)
        page.wait_for_timeout(600)
        click_at(page, "#compare", 0.55, 0.45, 1200)
        shot(page, "09_layers_zoom_ndsm_inspector.png")
        rep["layers"] = page.evaluate("() => [...document.querySelectorAll('.film')].map(f => f.dataset.id)")
        rep["readout2d"] = page.locator("#readout2d").inner_text()
        if a.reference:
            page.evaluate("() => document.querySelectorAll('.film[data-id=\"diff\"] .ab button')[1].click()")
            page.click("#reset-2d")
            page.wait_for_timeout(700)
            shot(page, "10_layers_reference_difference.png")

        # ---- 4. PNG: relative only, live
        page.keyboard.press("1")
        t0 = time.time()
        upload(page, a.png)
        page.wait_for_selector("#open-3d", state="hidden", timeout=30000)      # the new job has started
        page.wait_for_selector("#open-3d:not([hidden])", timeout=600000)
        rep["timings_s"]["png_job_wall"] = round(time.time() - t0, 1)
        page.wait_for_timeout(1200)
        shot(page, "11_processing_png_relative.png")
        rep["stages_png"] = page.evaluate(STAGES)
        page.click("#open-3d")
        page.wait_for_function("window.__dw.state.terrain !== null", timeout=60000)
        page.wait_for_timeout(2600)
        click_at(page, "#viewport", 0.55, 0.55, 1000)
        shot(page, "12_3d_png_relative.png")
        rep["png_stamp"] = page.locator("#stamp-title").inner_text()
        rep["fps_orbit_relative"] = orbit_fps(page)

        page.set_viewport_size({"width": 420, "height": 900})
        page.wait_for_timeout(800)
        shot(page, "13_narrow_layout.png", full_page=True)

        # blob: URLs are in-memory textures created by the glTF loader (same origin), not network requests
        rep["external_requests"] = sorted({u for u in requests if not u.startswith(("http://127.0.0.1:8000", "blob:http://127.0.0.1:8000"))})
        rep["page_errors"] = errors
        browser.close()
    (out / "ui_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: v for k, v in rep.items() if not k.startswith("stages")}, indent=2))


if __name__ == "__main__":
    main()
