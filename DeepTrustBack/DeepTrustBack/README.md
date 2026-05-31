# DeepTrustBack

Backend API built with **FastAPI**. It provides:

- **Media analysis** endpoints for image/video/audio deepfake detection (via external Hugging Face inference endpoints).
- **Logging** endpoints backed by a SQL database (`DetectionLog` table).

## Quickstart (local)

### 1) Create a virtual environment

```bash
python -m venv venv
```

### 2) Activate the virtual environment

macOS / Linux:

```bash
source venv/bin/activate
```

Windows (PowerShell):

```powershell
.\venv\Scripts\Activate.ps1
```

Windows (cmd):

```bat
venv\Scripts\activate.bat
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Configure environment variables

Create a `.env` file in the project root (or export env vars in your shell). At minimum you need:

- `DATABASE_URL` (**required**): SQLAlchemy connection string (Postgres recommended; SQLite also works for local dev).
- `HUGGINGFACE_API_KEY` (**required** for media endpoints)
- `HUGGINGFACE_IMAGE_API_URL` (**required** for image/video endpoints)
- `HUGGINGFACE_AUDIO_API_URL` (**required** for audio endpoint)

Example `.env.example`

### 5) Run the API server (uvicorn)

```bash
uvicorn app.app:app --host 0.0.0.0 --port 8000 --reload
```

Then open:

- Interactive docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/`

## Configuration (env vars)

### Required

- **`DATABASE_URL`**: required by `app/config/db.py`. The app will raise if it’s missing.
- **`HUGGINGFACE_API_KEY`**: required by `AudioAnalyzer`, `ImageAnalyzer`, `VideoAnalyzer`.
- **`HUGGINGFACE_IMAGE_API_URL`**: used by `ImageAnalyzer` and `VideoAnalyzer` (video calls the image endpoint per-frame).
- **`HUGGINGFACE_AUDIO_API_URL`**: used by `AudioAnalyzer`.

## External system dependencies

- **ffmpeg**: required to analyze non-WAV audio uploads (e.g. WebM/Opus from browsers). The code calls `ffmpeg` from `PATH`.
- **ffprobe** (optional): used for best-effort metadata/debugging in `media_controller.py`.

## API overview

### Logs (`app/controllers/log_controller.py`)

- **GET** `/logs/get_by_id?id=...`: fetch a single detection log row.
- **GET** `/logs/all`: list all detection logs.
- **GET** `/logs/by_state?state=deepfake|bonafide`: filter by classification.
- **DELETE** `/logs/delete_by_id?id=...`: delete a log row by id.

### Media (`app/controllers/media_controller.py`)

All are `multipart/form-data` with form field **`file`**.

- **POST** `/analyze_image`: calls the image inference endpoint and logs the result.
- **POST** `/analyze_video`: samples frames from the first N seconds, calls the image inference endpoint per-frame, aggregates the result, and logs it.
- **POST** `/analyze_audio`: calls the audio inference endpoint (converts to WAV first if needed), normalizes scoring, and logs it.

## Architecture (high level)

- **App entrypoint**: `app/app.py`
  - Creates the `FastAPI()` app
  - Enables permissive CORS
  - Includes routers: `log_handler`, `media_handler`
  - On startup, calls `init_db()` (best-effort; it won’t crash the server if DB init fails)

- **Controllers (API layer)**: `app/controllers/`
  - Define routes and HTTP inputs/outputs
  - Delegate work to services

- **Services (business logic)**: `app/services/`
  - `Analyzer` orchestrates `AudioAnalyzer`, `ImageAnalyzer`, `VideoAnalyzer`
  - `LogService` persists/queries the `DetectionLog` table

- **Models (DB layer)**: `app/models/`
  - SQLAlchemy table definition (`DetectionLog`)
  - `init_db()` creates tables and adds missing columns if needed

- **Schemas (DTO layer)**: `app/schemas/`
  - Pydantic models for API responses (e.g. `DetectionLog`)

- **Config**: `app/config/`
  - DB engine/session configuration (`db.py`)

- **Repository**: `app/repository/`
  - Currently empty/placeholder (the project uses `LogService` directly today)

## Directory structure

```text
DeepTrustBack/
  app/
    app.py
    __init__.py
    config/
      db.py
    controllers/
      log_controller.py
      media_controller.py
    core/
      security.py
    middleware/
    models/
      detection_log_model.py
    repository/
      detection_log_repository.py
    schemas/
      detection_log_schema.py
      other_schemas.py
    services/
      analyzer.py
      audio_analyzer.py
      image_analyzer.py
      log_service.py
      video_analyzer.py
    utils/
  temp_files/
    recording/
      ...
  tests/
  requirements.txt
  Procfile
```

## Deployment note (Procfile)

The repo includes a `Procfile` with:

```text
web: python -m uvicorn app.app:app --host 0.0.0.0 --port $PORT
```


