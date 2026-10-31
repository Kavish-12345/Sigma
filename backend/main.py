"""
FastAPI entry point.

Run from the project root:
    uvicorn backend.main:app --reload --port 7860
API docs: http://localhost:7860/docs
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")  # GROQ_API_KEY etc.

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
import json  # noqa: E402

from fastapi.responses import FileResponse, StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from backend.chat.form_state import normalise_form  # noqa: E402
from backend.chat.llm import chat_events, chat_turn, group_labels, question_spec  # noqa: E402
from backend.engine.options import options_payload  # noqa: E402
from backend.engine.recommend import load_engine, recommend  # noqa: E402

app = FastAPI(title="Sigma: AI-based material selection", version="0.3")
app.add_middleware(  # React dev server (Vite) runs on port 5173
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecommendRequest(BaseModel):
    part_name: Optional[str] = Field(None, examples=["Gear shaft"])
    function: str = Field("not_sure", examples=["shaft"])
    design_inputs: Optional[dict[str, Any]] = Field(None, examples=[{"P_kW": 10, "N_rpm": 960, "M_Nm": 150,
                                                                    "d_mm": 30, "fos": 2, "loading": "cyclic"}])
    properties: list[str] = Field(default_factory=lambda: ["not_sure"], examples=[["Heat-treatable", "Structural"]])
    primary_property: Optional[str] = Field(None, examples=["Heat-treatable"])
    environment: str = Field("not_sure", examples=["dry_indoor"])
    weight_priority: str = Field("not_sure", examples=["not_important"])
    cost_priority: str = Field("not_sure", examples=["important"])
    volume_cm3: Optional[float] = Field(None, examples=[120])
    min_limits: dict[str, float] = Field(default_factory=dict, examples=[{"Sy": 400}])
    include: list[str] = Field(default_factory=list, examples=[[]])
    avoid: list[str] = Field(default_factory=list, examples=[[]])
    part_notes: list[str] = Field(default_factory=list, examples=[[]])  # cautions from the chat, shown with the results
    price_overrides: dict[str, float] = Field(default_factory=dict, examples=[{}])
    top_n: int = Field(10, ge=1, le=50)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    form: Optional[dict[str, Any]] = None
    direct_answer: Optional[dict[str, Any]] = None  # from a question-card click


@app.on_event("startup")
def _warm_up():
    load_engine()  # load model, data and predictions once


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/options")
def get_options():
    eng = load_engine()
    return options_payload(eng["prices"]["categories"], eng["classes"])


@app.get("/api/prices")
def get_prices():
    return load_engine()["prices"]


@app.get("/api/materials/summary")
def materials_summary():
    """Read-only overview of the dataset for the homepage (stats + Ashby chart) and percentile bars."""
    eng = load_engine()
    df, meta = eng["df"], eng["meta"]
    cols = ["Material", "Heat treatment", "family", "Su", "Sy", "A5", "E", "Ro", "price_category"]
    rows = df[cols].astype(object).where(df[cols].notna(), None).to_dict(orient="records")
    return {
        "count": int(len(df)),
        "standards": sorted(df["Std"].dropna().unique().tolist()),
        "families": sorted(df["family"].unique().tolist()),
        "classes": eng["classes"],
        "model": {"name": meta.get("model"), "features": len(meta.get("features", [])),
                  "scores": meta.get("cv", {}).get("scores", {}), "n_train": meta.get("n_train")},
        "materials": rows,
    }


@app.post("/api/recommend")
def post_recommend(req: RecommendRequest):
    try:
        res = recommend(req.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    if not res["results"] and res.get("problem_fields"):
        # tell the UI which question to ask again (the chat re-asks it)
        form = normalise_form(req.model_dump())
        form["design_draft"] = dict(req.design_inputs or {})  # prefill the card with the previous values
        for f in res["problem_fields"]:
            form[f] = None
        eng = load_engine()
        res["question"] = question_spec(res["problem_fields"][0], group_labels(eng["prices"]["categories"]), form,
                                        eng["classes"])
    return res


@app.post("/api/chat")
def post_chat(req: ChatRequest):
    """Stateless: React sends the whole conversation and current form each turn."""
    eng = load_engine()
    return chat_turn([m.model_dump() for m in req.messages], req.form,
                     eng["prices"]["categories"], req.direct_answer, eng["classes"])


@app.post("/api/chat/stream")
def post_chat_stream(req: ChatRequest):
    """Same as /api/chat but streams the reply token by token (Server-Sent Events)."""
    eng = load_engine()
    cats, classes = eng["prices"]["categories"], eng["classes"]

    def gen():
        for ev in chat_events([m.model_dump() for m in req.messages], req.form, cats, req.direct_answer, classes):
            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---- Serve the built React app (frontend/dist) when it exists (Docker / deployment) ----
DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        f = DIST / path
        return FileResponse(f if path and f.is_file() else DIST / "index.html")
