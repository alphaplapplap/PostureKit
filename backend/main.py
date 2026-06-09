from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from utils.postgres_manager import ensure_postgres_running, stop_postgres

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    """Start PostgreSQL when FastAPI starts"""
    if not ensure_postgres_running():
        print("WARNING: PostgreSQL failed to start")

@app.on_event("shutdown")
async def shutdown_event():
    """Stop PostgreSQL when FastAPI stops"""
    stop_postgres()

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/detect")
async def detect_pose(file: UploadFile = File(...)):
    """Detect pose in uploaded image"""
    # Your PoseDetector logic here
    return {
        "detected": True,
        "confidence": 0.92,
        "keypoints": [...]
    }

@app.post("/search")
async def search_similar(
    file: UploadFile = File(...),
    k: int = 20,
    min_confidence: float = 0.5
):
    """Search for similar poses"""
    # Your SimilarityEngine logic here
    return {
        "results": [
            {
                "id": "uuid",
                "similarity": 98,
                "filename": "yoga_001.jpg",
                "confidence": 0.92,
                "image_path": "/path/to/image.jpg"
            }
        ],
        "search_time": 0.34
    }

@app.post("/index/start")
async def start_indexing(
    directory: str,
    recursive: bool = True,
    min_confidence: float = 0.3
):
    """Start directory indexing"""
    # Your StorageManager + FAISS indexing logic
    return {"task_id": "uuid"}

@app.get("/index/status/{task_id}")
async def get_index_status(task_id: str):
    """Get indexing progress"""
    return {
        "progress": 0.45,
        "current_file": "yoga_045.jpg",
        "poses_indexed": 556,
        "images_processed": 235,
        "total_images": 523
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
