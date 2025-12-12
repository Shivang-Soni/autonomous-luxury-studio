from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes import router as api_router
from config import Configuration

config = Configuration()

app = FastAPI(
    title="Autonomous Luxury Studio API",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Register API routes
app.include_router(api_router)


@app.get("/health")
def health_check():
    return {"status": "OK"}
