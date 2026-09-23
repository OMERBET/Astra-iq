"""
ASTRA — AI Space Sentinel
Single FastAPI app deployed as one Vercel Python Function.
Vercel auto-detects this file (api/index.py) and routes every
/api/* request to it — no vercel.json needed.
"""

import os
import sys
import traceback

# Vercel's Python runtime doesn't automatically add this file's own
# directory to sys.path, so a plain "import gee_engine" fails in
# production even though it works locally. Add it explicitly.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import gee_engine as engine

app = FastAPI(title="ASTRA — AI Space Sentinel")


@app.exception_handler(Exception)
async def debug_exception_handler(request: Request, exc: Exception):
    # TEMPORARY: surfaces the real Python error in the browser while we
    # debug the deployment. Remove/replace with a generic message once
    # everything is confirmed working.
    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "type": type(exc).__name__,
            "traceback": traceback.format_exc().splitlines()[-15:],
        },
    )


_ee_ready = False


def _ensure_ee():
    global _ee_ready
    if not _ee_ready:
        engine.init_ee()
        _ee_ready = True


class AnalyzeRequest(BaseModel):
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float
    before_start: str = Field(..., description="YYYY-MM-DD")
    before_end: str = Field(..., description="YYYY-MM-DD")
    after_start: str = Field(..., description="YYYY-MM-DD")
    after_end: str = Field(..., description="YYYY-MM-DD")
    index: str = Field("NDVI", description="NDVI or NDWI")


@app.get("/api/health")
def health():
    try:
        _ensure_ee()
        return {"status": "ok", "earth_engine_ready": True}
    except Exception as e:
        return {"status": "error", "earth_engine_ready": False, "detail": str(e)}


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    if req.index not in ("NDVI", "NDWI"):
        raise HTTPException(400, "index must be NDVI or NDWI")

    _ensure_ee()
    region = engine.make_region(req.min_lon, req.min_lat, req.max_lon, req.max_lat)

    try:
        result = engine.detect_change(
            region=region,
            before_start=req.before_start,
            before_end=req.before_end,
            after_start=req.after_start,
            after_end=req.after_end,
            index=req.index,
        )
    except Exception as e:
        raise HTTPException(500, f"analysis failed: {e}")

    return {
        "region": {
            "min_lon": req.min_lon,
            "min_lat": req.min_lat,
            "max_lon": req.max_lon,
            "max_lat": req.max_lat,
        },
        **result,
    }
