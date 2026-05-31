from app.services.analyzer import Analyzer
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.encoders import jsonable_encoder
from app.services.log_service import LogService
from datetime import date, datetime
from pydantic import BaseModel, Field
import json
import os
import subprocess
import tempfile


def _sniff_media_format(file_bytes: bytes, filename: str = "", content_type: str = "") -> str:
    """
    Best-effort "what kind of audio/video is this?" detector based on:
    - provided content_type
    - filename extension
    - magic bytes
    """
    ext = (os.path.splitext(filename or "")[1] or "").lower().lstrip(".")
    head = file_bytes[:16] if isinstance(file_bytes, (bytes, bytearray)) else b""

    # Common magic numbers
    if head.startswith(b"RIFF") and file_bytes[8:12] == b"WAVE":
        magic = "wav (RIFF/WAVE)"
    elif head.startswith(b"fLaC"):
        magic = "flac"
    elif head.startswith(b"OggS"):
        magic = "ogg/opus (OggS container)"
    elif head.startswith(b"ID3"):
        magic = "mp3 (ID3 tag)"
    elif head[:2] == b"\xff\xfb" or head[:2] == b"\xff\xf3" or head[:2] == b"\xff\xf2":
        magic = "mp3 (frame sync)"
    elif head.startswith(b"\x00\x00\x00") and b"ftyp" in file_bytes[:16]:
        magic = "mp4/m4a (ISO BMFF ftyp)"
    elif head.startswith(b"\x1a\x45\xdf\xa3"):
        magic = "webm/mkv (EBML)"
    else:
        magic = "unknown"

    guessed_by_ext = ""
    if ext:
        guessed_by_ext = f"ext={ext}"

    guessed_by_mime = ""
    if content_type:
        guessed_by_mime = f"content_type={content_type}"

    return " | ".join([p for p in [magic, guessed_by_ext, guessed_by_mime] if p]) or "unknown"


def _ffprobe_summary(file_bytes: bytes, filename: str = "") -> dict:
    """
    Best-effort ffprobe metadata (if ffprobe is installed).
    Returns a dict with either {"ffprobe": <json>} or {"ffprobe_error": "..."}.
    """
    suffix = os.path.splitext(filename or "")[1] or ""
    try:
        with tempfile.NamedTemporaryFile(prefix="deeptrust_", suffix=suffix, delete=True) as tmp:
            tmp.write(file_bytes)
            tmp.flush()

            cmd = [
                "ffprobe",
                "-hide_banner",
                "-loglevel",
                "error",
                "-show_format",
                "-show_streams",
                "-of",
                "json",
                tmp.name,
            ]
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
            return {"ffprobe": json.loads(out)}
    except FileNotFoundError:
        return {"ffprobe_error": "ffprobe not installed / not in PATH"}
    except subprocess.CalledProcessError as e:
        return {"ffprobe_error": f"ffprobe failed: exit={e.returncode} output={str(e.output)[:800]}"}
    except Exception as e:
        return {"ffprobe_error": f"ffprobe unexpected error: {type(e).__name__}: {e}"}


def _clamp01(x: float) -> float:
    try:
        return max(0.0, min(float(x), 1.0))
    except Exception:
        return 0.0


def _extract_label_to_score(output) -> dict[str, float]:
    """
    Normalize common HF-style image classification outputs into {label_lower: score_float}.

    Supports:
    - list[{"label": str, "score": float}, ...]
    - {"outputs": [...]} wrapper
    - {"label": "...", "score": ...} singleton
    """
    try:
        items = output
        if isinstance(output, dict):
            if "outputs" in output and isinstance(output["outputs"], list):
                items = output["outputs"]
            elif "label" in output and "score" in output:
                items = [output]

        if not isinstance(items, list):
            return {}

        out: dict[str, float] = {}
        for it in items:
            if not isinstance(it, dict):
                continue
            label = str(it.get("label", "")).strip()
            score = it.get("score", None)
            try:
                score_f = float(score)
            except Exception:
                continue
            if label:
                out[label.strip().lower()] = score_f
        return out
    except Exception:
        return {}


def _get_realism_score_01(label_scores: dict[str, float]) -> float | None:
    """
    Pull a "realness" score from the model output.
    Different image deepfake models use different label names.
    """
    for key in ("realism", "real", "bonafide", "bona fide"):
        if key in label_scores:
            s = label_scores[key]
            try:
                return max(0.0, min(float(s), 1.0))
            except Exception:
                return None
    return None


def _get_deepfake_prob_01(label_scores: dict[str, float]) -> float | None:
    """
    Extract a "deepfake probability" (0..1) from the model output.

    Supported patterns:
    - explicit fake label: "deepfake", "fake", "spoof", "manipulated", "synthetic"
    - explicit real label: "realism", "real", "bonafide", "bona fide"  -> deepfake_prob = 1 - real_prob

    Returns None if nothing interpretable is present.
    """
    if not label_scores:
        return None

    # 1) Prefer explicit deepfake-ish labels if present.
    deepfake_keys = ("deepfake", "fake", "spoof", "manipulated", "synthetic")
    candidates = []
    for k, v in label_scores.items():
        lk = (k or "").strip().lower()
        if any(tok in lk for tok in deepfake_keys):
            candidates.append(_clamp01(v))
    if candidates:
        # If multiple "fake-ish" labels exist, use the strongest signal.
        return max(candidates)

    # 2) Fall back to explicit "realness" label -> invert.
    realism = _get_realism_score_01(label_scores)
    if realism is not None:
        return _clamp01(1.0 - realism)

    return None


def _classify_from_deepfake_prob(deepfake_prob_01: float, *, threshold_0100: float = 50.0) -> tuple[str, float, float]:
    """
    Convert deepfake probability (0..1) into:
    - classification: "Bonafide" | "Deepfake"
    - score: deepfake risk score (0..100), higher = more likely deepfake
    - fidelity: confidence in the decision (0..100), aligned with the audio model confidence semantics:
      - 0   => p=0.5 (borderline / guessing)
      - 100 => p=0.0 or p=1.0 (max certainty)
      Computed as `abs(p - 0.5) * 2 * 100`.
    """
    p = _clamp01(deepfake_prob_01)
    score_0100 = p * 100.0
    classification = "Deepfake" if score_0100 >= float(threshold_0100) else "Bonafide"
    fidelity_0100 = abs(p - 0.5) * 2.0 * 100.0
    return classification, score_0100, fidelity_0100


class AudioAnalysisResponse(BaseModel):
    """
    Output payload for `POST /analyze_audio`.

    This endpoint intentionally returns a *minimal* response so the client does not need to
    understand the full inference payload.

    - **classification**: final decision label.
      - `"Bonafide"` = real/human audio
      - `"Deepfake"` = spoofed / manipulated audio
    - **score**: deepfake risk score in range `0..100` (higher = more likely deepfake).
      For audio, this is derived from the model's raw `deepfake_score` (`0..2`) and its
      `threshold_used` (default 0.5) such that **50 is exactly the decision boundary**.
    - **fidelity**: confidence in the predicted class in range `0..100`.
      For audio, this is taken from the model's own `confidence` field (0..1) when present.
    """

    classification: str = Field(..., examples=["Bonafide", "Deepfake"])
    score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description=(
            "Deepfake risk score (0..100). Higher = more likely deepfake. "
            "For audio, mapped from raw deepfake_score (0..2) and threshold_used (default 0.5) "
            "so that score=50 corresponds exactly to the decision boundary."
        ),
        examples=[0.7, 45.2, 98.9],
    )
    fidelity: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Confidence in the chosen class (0..100). For audio, derived from the model's `confidence` (0..1).",
        examples=[0.0, 40.0, 95.0],
    )


class VideoAnalysisResponse(BaseModel):
    """
    Output payload for `POST /analyze_video`.

    The underlying video pipeline samples frames and calls an image inference endpoint.
    This endpoint returns a minimal summary, similar to `/analyze_audio`.
    """

    classification: str = Field(..., examples=["Bonafide", "Deepfake"])
    score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description=(
            "Mean deepfake risk score (0..100) aggregated across up to 10 sampled frames. "
            "Higher = more likely deepfake."
        ),
        examples=[12.0, 55.0, 97.0],
    )
    fidelity: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description=(
            "Confidence in the decision (0..100). 0 means borderline/uncertain, 100 means very certain. "
            "Computed from the mean deepfake probability."
        ),
        examples=[0.0, 22.0, 96.0],
    )


class ImageAnalysisResponse(BaseModel):
    """
    Output payload for `POST /analyze_image`.

    Uses the same underlying image inference endpoint as the video pipeline (per-frame inference),
    but runs it once for the uploaded image.
    """

    classification: str = Field(..., examples=["Bonafide", "Deepfake"])
    score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Deepfake risk score (0..100) derived from the image model output. Higher = more likely deepfake.",
        examples=[8.0, 42.0, 93.5],
    )
    fidelity: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Confidence in the decision (0..100). 0 means borderline/uncertain, 100 means very certain.",
        examples=[0.0, 32.0, 90.0],
    )


media_handler = APIRouter()

@media_handler.post(
    "/analyze_image",
    tags=["media"],
    summary="Analyze an image for deepfake signals",
    description=(
        "## Input\n"
        "- **Content-Type**: `multipart/form-data`\n"
        "- **Form field**: `file` (UploadFile)\n\n"
        "## What this endpoint does\n"
        "1. Reads the uploaded image bytes.\n"
        "2. Calls the configured image inference endpoint (same endpoint used by `/analyze_video`).\n"
        "3. Derives a classification + score.\n"
        "4. Stores a detection log entry.\n\n"
        "## Output\n"
        "Returns a minimal JSON payload:\n"
        "```json\n"
        "{\"classification\": \"Bonafide\", \"score\": 28.0, \"fidelity\": 72.0}\n"
        "```"
    ),
    response_model=ImageAnalysisResponse,
    responses={200: {"description": "Classification + score + fidelity."}, 502: {"description": "Upstream inference endpoint error."}},
)
async def post_image(
    file: UploadFile = File(..., description="Image file to analyze (multipart/form-data field name: `file`).")
):
    analyzer = Analyzer()
    image_data = await file.read()

    try:
        result = analyzer.analyze_image(
            image_data,
            filename=getattr(file, "filename", None),
            content_type=getattr(file, "content_type", None),
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Image inference failed: {type(e).__name__}: {e}")

    if isinstance(result, dict) and "error" in result:
        raise HTTPException(status_code=502, detail=result["error"])

    label_scores = _extract_label_to_score(result)
    deepfake_prob_01 = _get_deepfake_prob_01(label_scores)
    if deepfake_prob_01 is None:
        raise HTTPException(status_code=502, detail="Image inference succeeded but no parsable deepfake probability was returned.")

    threshold = float(os.getenv("IMAGE_DEEPFAKE_THRESHOLD_0100", "50.0"))
    classification, score_0100, fidelity_0100 = _classify_from_deepfake_prob(deepfake_prob_01, threshold_0100=threshold)
    is_deepfake = classification == "Deepfake"

    log = {
        "isDeepFake": is_deepfake,
        "date": date.today(),
        "hour": datetime.now().time(),
        "classification": classification,
        "score": score_0100,
        "fidelity": fidelity_0100,
    }

    log_service = LogService()
    log_service.save_log(log)

    return jsonable_encoder({"classification": classification, "score": score_0100, "fidelity": fidelity_0100})


@media_handler.post(
    "/analyze_video",
    tags=["media"],
    summary="Analyze a video for deepfake signals",
    description=(
        "## Input\n"
        "- **Content-Type**: `multipart/form-data`\n"
        "- **Form field**: `file` (UploadFile)\n\n"
        "## What this endpoint does\n"
        "1. Reads the uploaded video bytes.\n"
        "2. Samples frames from the first seconds of the video and calls the configured image inference endpoint.\n"
        "3. Derives a classification + score.\n"
        "4. Stores a detection log entry.\n\n"
        "## Output\n"
        "Returns a minimal JSON payload:\n"
        "```json\n"
        "{\"classification\": \"Bonafide\", \"score\": 28.0, \"fidelity\": 72.0}\n"
        "```"
    ),
    response_model=VideoAnalysisResponse,
    responses={200: {"description": "Classification + score + fidelity."}, 502: {"description": "Upstream inference endpoint error."}},
)
async def post_video(
    file: UploadFile = File(..., description="Video file to analyze (multipart/form-data field name: `file`).")
):
    """
    Video analysis endpoint.

    - **Input**: multipart/form-data file upload (the video)
    - **Output**: (future) model inference result
    - **Current behavior**: returns 501 until video inference is implemented
    """
    
    analyzer = Analyzer()

    video_data = await file.read()
    
    try:
        # Sample enough frames to make the final decision more stable.
        result = analyzer.analyze_video(video_data, filename=getattr(file, "filename", None), frames=10, seconds=10)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Video inference failed: {type(e).__name__}: {e}")

    if isinstance(result, dict) and "error" in result:
        raise HTTPException(status_code=502, detail=result["error"])

    # ---- Derive score + classification from per-frame image endpoint outputs ----
    # We extract a per-frame deepfake probability (0..1) and average over up to 10 frames.
    threshold = float(os.getenv("VIDEO_DEEPFAKE_THRESHOLD_0100", "50.0"))

    frame_results = []
    if isinstance(result, dict):
        frame_results = result.get("per_frame_results") or []

    deepfake_probs: list[float] = []
    for fr in frame_results:
        if not isinstance(fr, dict):
            continue
        out = fr.get("output")
        label_scores = _extract_label_to_score(out)
        p = _get_deepfake_prob_01(label_scores)
        if p is not None:
            deepfake_probs.append(p)

    if not deepfake_probs:
        # Nothing parsable; avoid logging misleading score.
        raise HTTPException(status_code=502, detail="Video inference succeeded but no parsable frame probabilities were returned.")

    # Mean of up to 10 deepfake probabilities (0..1), then scale to 0..100.
    probs_for_mean = deepfake_probs[:10]
    mean_deepfake_prob = sum(probs_for_mean) / float(len(probs_for_mean))
    classification, score_0100, fidelity_0100 = _classify_from_deepfake_prob(mean_deepfake_prob, threshold_0100=threshold)
    is_deepfake = classification == "Deepfake"

    # ---- Persist log (same pattern as audio) ----
    log = {
        "isDeepFake": is_deepfake,
        "date": date.today(),
        "hour": datetime.now().time(),
        "classification": classification,
        "score": score_0100,
        "fidelity": fidelity_0100,
    }

    log_service = LogService()
    log_service.save_log(log)

    return jsonable_encoder(
        {
            "classification": classification,
            "score": score_0100,
            "fidelity": fidelity_0100,
        }
    )


@media_handler.post(
    "/analyze_audio",
    tags=["media"],
    summary="Analyze an audio clip and log the result",
    description=(
        "## Input\n"
        "- **Content-Type**: `multipart/form-data`\n"
        "- **Form field**: `file` (UploadFile)\n\n"
        "## What this endpoint does\n"
        "1. Reads the uploaded audio bytes.\n"
        "2. Calls the configured inference endpoint (via `AudioAnalyzer`).\n"
        "3. Normalizes the raw `deepfake_score` (0..2) into a client-friendly `score` (0..100).\n"
        "4. Computes a `fidelity` value (0..100) representing confidence in the predicted class.\n"
        "5. Saves a log entry to the `DetectionLog` table with:\n"
        "   - `classification` (\"Bonafide\" | \"Deepfake\")\n"
        "   - `score` (0..100)\n"
        "   - `fidelity` (0..100)\n"
        "   - `date`, `hour` (server time)\n\n"
        "## Output\n"
        "Returns a minimal JSON payload:\n"
        "```json\n"
        "{\"classification\": \"Bonafide\", \"score\": 25.0, \"fidelity\": 80.0}\n"
        "```\n\n"
        "## Scoring / Normalization\n"
        "- Raw `deepfake_score` is expected in range **0..2** (higher = more fake)\n"
        "- Raw `threshold_used` is expected in range **0..2** (default: **0.5**)\n"
        "- Returned `score` is a deepfake risk score **0..100** with **50 at the decision boundary**:\n"
        "  - if `deepfake_score <= threshold_used`: `score = clamp(deepfake_score/threshold_used,0..1) * 50`\n"
        "  - else: `score = 50 + clamp((deepfake_score-threshold_used)/(2-threshold_used),0..1) * 50`\n"
        "- Returned `fidelity` is `confidence * 100` when the model returns `confidence`.\n\n"
        "## Classification\n"
        "- Uses `is_bonafide` as the primary decision flag when present.\n"
    ),
    response_model=AudioAnalysisResponse,
    responses={
        200: {"description": "Classification + normalized score + fidelity."},
        502: {"description": "Upstream inference endpoint error."},
    },
)
async def post_audio(
    file: UploadFile = File(..., description="Audio file to analyze (multipart/form-data field name: `file`).")
):
    """
    Audio analysis endpoint.

    - **Input**: multipart/form-data file upload (the audio clip)
    - **Side effects**: writes a row into the `DetectionLog` table with:
      - `classification` ("Bonafide" | "Deepfake")
      - `score` (0..100)
      - `date`, `hour` (server time)
    - **Output**: minimal JSON response with `classification` and `score`
    """
    
    analyzer = Analyzer()
    audio_data = await file.read()

    # ---- Call analyzer ----
    try:
        result = analyzer.analyze_audio(
            audio_data,
            filename=getattr(file, "filename", None),
            content_type=getattr(file, "content_type", None),
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Audio inference failed: {type(e).__name__}: {e}")
    
    if isinstance(result, dict) and "error" in result:
        raise HTTPException(status_code=502, detail=result["error"])

    # Audio endpoint contract (as provided):
    # - is_bonafide: primary decision flag
    # - deepfake_score: 0..2, with threshold_used (default 0.5) as the decision boundary
    # - confidence: 0..1 expressing certainty in the decision
    #
    # We map deepfake_score -> score_0100 with the invariant:
    #   deepfake_score == threshold_used  => score_0100 == 50
    raw_score = float(result.get("deepfake_score", 0.0) or 0.0)
    threshold_used = float(result.get("threshold_used", 0.5) or 0.5)
    max_raw = float(os.getenv("AUDIO_DEEPFAKE_SCORE_MAX", "2.0"))

    # Guard against bad configs from upstream.
    if threshold_used <= 0.0:
        threshold_used = 0.5
    if max_raw <= threshold_used:
        max_raw = max(2.0, threshold_used + 1e-6)

    # Piecewise-linear mapping to 0..100 with 50 at the boundary.
    if raw_score <= threshold_used:
        # 0..threshold -> 0..50
        score_0100 = _clamp01(raw_score / threshold_used) * 50.0
    else:
        # threshold..max_raw -> 50..100 (saturate at max_raw)
        score_0100 = 50.0 + _clamp01((raw_score - threshold_used) / (max_raw - threshold_used)) * 50.0

    # Classification: trust the explicit decision flag when present.
    is_bonafide = result.get("is_bonafide", None)
    if is_bonafide is None:
        label = str(result.get("label", "")).strip().lower()
        if label in ("bonafide", "bona fide", "real"):
            is_bonafide = True
        elif label in ("spoof", "deepfake", "fake"):
            is_bonafide = False
        else:
            # Fallback: use score boundary.
            is_bonafide = bool(score_0100 < 50.0)

    classification = "Bonafide" if bool(is_bonafide) else "Deepfake"
    is_deepfake = classification == "Deepfake"  # keep boolean for backward compatibility

    # Fidelity: prefer model-provided confidence when available.
    conf = result.get("confidence", None)
    if conf is not None:
        fidelity_0100 = _clamp01(float(conf)) * 100.0
    else:
        p = _clamp01(score_0100 / 100.0)
        fidelity_0100 = abs(p - 0.5) * 2.0 * 100.0

    log = {
        "isDeepFake": is_deepfake,
        "date": date.today(),
        "hour": datetime.now().time(),
        "classification": classification,
        "score": score_0100,
        "fidelity": fidelity_0100,
    }
        
    log_service = LogService()
    
    log_service.save_log(log)
    
    return jsonable_encoder(
        {
            "classification": classification,
            "score": score_0100,
            "fidelity": fidelity_0100,
        }
    )
