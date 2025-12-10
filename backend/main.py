from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes import router as api_router
from backend.config import Configuration

config = Configuration()


app = FastAPI(
    title="Autonomous Luxury Studio API",
    version="1.0.0"
)

config = Configuration()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Register routes
app.include(api_router)


@app.get("/health")
def health_check():
    return {"status": "OK"}