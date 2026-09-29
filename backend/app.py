"""DepthWizard local API (FastAPI). The ML engine lives in depthwizard.pipeline; this only queues and serves.

  PYTHONPATH=src:. uvicorn backend.app:app --host 127.0.0.1 --port 8000

POST /jobs (multipart: file, model, calibration, dem_source, band_order) -> {id, status}
GET  /jobs, /jobs/{id}                  status, progress, metadata when done
GET  /jobs/{id}/files/{name}            products (SERVED) when done; stage previews (layer_*.png, calib_*.png,
                                        layers.json) also while running
GET  /jobs/{id}/events?after=N          pipeline stage events (JSON)
GET  /jobs/{id}/stream                  the same events as server-sent events, live, then 'end'
GET  /jobs/{id}/point?row=&col=  or ?x=&y=   elevation (+ DERIVED height above an estimated ground, with confidence)
GET  /health
Jobs run one at a time in a background worker (single 6 GB GPU).
"""
from __future__ import annotations

import asyncio
import json
import os
import queue
import re
import shutil
import threading
import traceback
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from depthwizard.pipeline import BAND_ORDERS, CALIBRATIONS, DEM_SOURCES, PipelineConfig, compare_reference, query_point, run_pipeline

ALLOWED_EXT = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
SERVED = {"dsm.tif", "rdsm.tif", "dem.tif", "ground.tif", "ndsm.tif", "uncertainty.tif", "slope.tif",
          "height_confidence.tif", "mesh.glb", "mesh_lod1.glb", "texture.jpg", "slope.png",
          "metadata.json", "calibration.json", "reference_diff.png", "reference_metrics.json"}
# stage previews are served while the job runs (live thumbnails); names are fixed patterns, never paths
LIVE_RE = re.compile(r"^(layer_[a-z0-9_]{1,40}\.png|calib_(scatter|residuals)\.png|layers\.json)$")
FRONTEND = Path(__file__).resolve().parents[1] / "frontend"
MODELS = ("da3_mono_large", "da2_small")
MAX_UPLOAD_MB = int(os.environ.get("DW_MAX_UPLOAD_MB", "2048"))
ID_RE = re.compile(r"^[0-9a-f]{32}$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobManager:
    def __init__(self, jobs_dir: Path, base_cfg: PipelineConfig, overrides: dict[str, Any] | None = None):
        self.dir = Path(jobs_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.base_cfg, self.overrides = base_cfg, overrides or {}
        self.q: queue.Queue[str] = queue.Queue()
        self.lock = threading.Lock()
        self.jobs: dict[str, dict] = {}
        self.events: dict[str, list[dict]] = {}
        for jf in self.dir.glob("*/job.json"):  # reload finished/failed jobs across restarts
            j = json.loads(jf.read_text())
            if j["status"] in ("queued", "running"):
                j.update(status="failed", error="server restarted while the job was pending")
            self.jobs[j["id"]] = j
        threading.Thread(target=self._worker, daemon=True).start()

    def _save(self, j: dict) -> None:
        (self.dir / j["id"] / "job.json").write_text(json.dumps(j, indent=2, default=str))

    def update(self, jid: str, **kw) -> None:
        with self.lock:
            self.jobs[jid].update(kw, updated=_now())
            self._save(self.jobs[jid])

    def add_event(self, jid: str, ev: dict) -> None:
        """Pipeline stage event (stage_start / stage_done / stage_skipped / stage_failed / done), numbered."""
        with self.lock:
            lst = self.events.setdefault(jid, [])
            ev = {"seq": len(lst), **json.loads(json.dumps(ev, default=str))}
            lst.append(ev)
            with open(self.dir / jid / "events.jsonl", "a") as f:
                f.write(json.dumps(ev) + "\n")

    def get_events(self, jid: str, after: int = -1) -> list[dict]:
        if jid not in self.events:                           # finished before a restart: read from disk
            p = self.dir / jid / "events.jsonl"
            self.events[jid] = [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
        return [e for e in self.events[jid] if e["seq"] > after]

    def submit(self, input_path: Path, params: dict) -> dict:
        jid = input_path.parent.name
        j = {"id": jid, "status": "queued", "progress": 0.0, "message": "queued", "params": params,
             "input": input_path.name, "created": _now(), "updated": _now(), "error": None, "metadata": None}
        with self.lock:
            self.jobs[jid] = j
            self._save(j)
        self.q.put(jid)
        return j

    def _worker(self) -> None:
        while True:
            jid = self.q.get()
            j = self.jobs[jid]
            cfg = replace(self.base_cfg, **j["params"])
            self.update(jid, status="running", message="starting")
            try:
                md = run_pipeline(self.dir / jid / j["input"], self.dir / jid, cfg,
                                  predictor=self.overrides.get("predictor"),
                                  dem_source=self.overrides.get("dem_source"),
                                  progress=lambda m, f: self.update(jid, message=m, progress=round(f, 3)),
                                  emit=lambda ev: self.add_event(jid, ev))
                self.update(jid, status="done", progress=1.0, message="done", metadata=md)
            except Exception as exc:  # noqa: BLE001 - report every failure to the client
                self.update(jid, status="failed", message="failed", error=f"{type(exc).__name__}: {exc}",
                            traceback=traceback.format_exc()[-3000:])
            finally:
                self.q.task_done()


def create_app(jobs_dir: str | Path = "runs/jobs", base_cfg: PipelineConfig | None = None,
               overrides: dict[str, Any] | None = None) -> FastAPI:
    app = FastAPI(title="DepthWizard", version="0.1.0")
    jm = JobManager(Path(jobs_dir), base_cfg or PipelineConfig(), overrides)
    app.state.jobs = jm

    def get_job(jid: str) -> dict:
        if not ID_RE.match(jid) or jid not in jm.jobs:
            raise HTTPException(404, "no such job")
        return jm.jobs[jid]

    @app.get("/health")
    def health():
        import torch
        return {"ok": True, "cuda": torch.cuda.is_available(), "queued": jm.q.qsize(),
                "models": MODELS, "calibrations": CALIBRATIONS, "dem_sources": list(DEM_SOURCES)}

    @app.post("/jobs", status_code=202)
    def create_job(file: UploadFile = File(...), model: str = Form("da3_mono_large"),
                   calibration: str = Form("dem_plus_smooth_residual"), dem_source: str = Form("copernicus"),
                   band_order: str = Form("auto")):
        ext = Path(file.filename or "").suffix.lower()
        if ext not in ALLOWED_EXT:
            raise HTTPException(415, f"unsupported file type {ext!r}; use {sorted(ALLOWED_EXT)}")
        if model not in MODELS or calibration not in CALIBRATIONS or dem_source not in DEM_SOURCES:
            raise HTTPException(422, "bad model / calibration / dem_source")
        if band_order != "auto" and band_order not in BAND_ORDERS and not re.fullmatch(r"\d{1,2},\d{1,2},\d{1,2}", band_order):
            raise HTTPException(422, f"band_order must be auto, {sorted(BAND_ORDERS)} or 'r,g,b' band numbers")
        jid = uuid.uuid4().hex
        d = jm.dir / jid
        d.mkdir(parents=True)
        dst = d / f"input{ext}"
        with open(dst, "wb") as f:
            shutil.copyfileobj(file.file, f, length=1 << 20)
        if dst.stat().st_size > MAX_UPLOAD_MB * 1e6:
            shutil.rmtree(d)
            raise HTTPException(413, f"file larger than {MAX_UPLOAD_MB} MB")
        j = jm.submit(dst, {"model": model, "calibration": calibration, "dem_source": dem_source,
                            "band_order": band_order})
        jm.update(jid, original_name=Path(file.filename or dst.name).name[:200])
        return {"id": jid, "status": j["status"]}

    @app.get("/jobs")
    def list_jobs():
        return [{**{k: j[k] for k in ("id", "status", "progress", "created", "input")},
                 "name": j.get("original_name") or j["input"]} for j in jm.jobs.values()]

    @app.get("/jobs/{jid}")
    def job_status(jid: str):
        return {k: v for k, v in get_job(jid).items() if k != "traceback"}

    @app.get("/jobs/{jid}/events")
    def job_events(jid: str, after: int = -1):
        j = get_job(jid)
        return {"status": j["status"], "error": j.get("error"), "events": jm.get_events(jid, after)}

    @app.get("/jobs/{jid}/stream")
    async def job_stream(jid: str, request: Request):
        """Server-sent events: every pipeline event as it happens, then an 'end' event. Resumes after
        Last-Event-ID (EventSource reconnects send it)."""
        get_job(jid)
        last = int(request.headers.get("last-event-id", "-1") or -1)

        async def gen():
            nonlocal last
            idle = 0
            while True:
                if await request.is_disconnected():
                    return
                evs = jm.get_events(jid, last)
                for ev in evs:
                    last = ev["seq"]
                    yield f"id: {ev['seq']}\ndata: {json.dumps(ev)}\n\n"
                j = jm.jobs[jid]
                if j["status"] in ("done", "failed") and not jm.get_events(jid, last):
                    yield f"event: end\ndata: {json.dumps({'status': j['status'], 'error': j.get('error')})}\n\n"
                    return
                idle = 0 if evs else idle + 1
                if idle and idle % 60 == 0:
                    yield ": keep-alive\n\n"
                await asyncio.sleep(0.25)
        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/jobs/{jid}/files/{name}")
    def job_file(jid: str, name: str):
        j = get_job(jid)
        live = bool(LIVE_RE.match(name))
        if name not in SERVED and not live:
            raise HTTPException(404, "unknown product")
        p = jm.dir / jid / name
        if (j["status"] != "done" and not live) or not p.exists():
            raise HTTPException(404, f"{name} not available (job {j['status']})")
        return FileResponse(p, filename=name, headers={"Cache-Control": "no-store"} if name == "layers.json" else None)

    @app.post("/jobs/{jid}/reference")
    def job_reference(jid: str, file: UploadFile = File(...)):
        j = get_job(jid)
        if j["status"] != "done":
            raise HTTPException(409, "job not finished")
        if Path(file.filename or "").suffix.lower() not in {".tif", ".tiff"}:
            raise HTTPException(415, "reference must be a GeoTIFF (.tif/.tiff)")
        dst = jm.dir / jid / "reference.tif"
        with open(dst, "wb") as f:
            shutil.copyfileobj(file.file, f, length=1 << 20)
        try:
            return compare_reference(jm.dir / jid, dst)
        except ValueError as exc:
            raise HTTPException(422, str(exc))

    @app.get("/jobs/{jid}/point")
    def job_point(jid: str, row: int | None = None, col: int | None = None,
                  x: float | None = None, y: float | None = None):
        j = get_job(jid)
        if j["status"] != "done":
            raise HTTPException(409, "job not finished")
        try:
            return query_point(jm.dir / jid, row, col, x, y)
        except (ValueError, IndexError) as exc:
            raise HTTPException(422, str(exc))

    if FRONTEND.exists():
        app.mount("/app", StaticFiles(directory=FRONTEND, html=True), name="app")

        @app.get("/", include_in_schema=False)
        def root():
            return RedirectResponse("/app/")

    return app


app = create_app(os.environ.get("DW_JOBS_DIR", "runs/jobs"))
