from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.controllers.log_controller import log_handler
from app.controllers.media_controller import media_handler

app = FastAPI()


@app.on_event("startup")  # pyright: ignore[reportDeprecated]
async def startup():
    # Startup: initialize database (sync). Must not terminate the process.
    try:
        from app.models.detection_log_model import init_db
        init_db()
    except Exception as e:
        # Don't fail startup - let the app start and handle DB errors at runtime.
        print(f"Database initialization warning: {e}")

# Configure CORS to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(log_handler)
app.include_router(media_handler)

@app.get("/")
def root():
    return {"Hello": "World"}
