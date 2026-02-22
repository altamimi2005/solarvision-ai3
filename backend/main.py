import os
import uuid
from typing import List, Dict
from fastapi import FastAPI, UploadFile, File, Response, Cookie
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from .inference import process_upload

app = FastAPI(title="SolarVision AI Backend")

@app.on_event("startup")
async def startup_event():
    print("------------------------------------------")
    print("SERVER IS STARTING UP AND READY!")
    print("------------------------------------------")

# Allowing CORS (optional but good for local dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session storage (Session ID -> List of result dictionaries)
session_results: Dict[str, List[dict]] = {}

@app.post("/api/upload")
async def upload_images(
    response: Response, 
    session_id: str = Cookie(None),
    files: List[UploadFile] = File(...)
):
    # Setup session tracking if no session ID provided
    if not session_id:
        session_id = str(uuid.uuid4())
        response.set_cookie(key="session_id", value=session_id, httponly=True)
        session_results[session_id] = []
    elif session_id not in session_results:
        session_results[session_id] = []
        
    # Clear previous results for this session if we want to overwrite
    session_results[session_id] = []
    
    results = []
    for file in files:
        contents = await file.read()
        
        try:
            res = process_upload(contents, file.filename)
            results.append(res)
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": f"Processing failed: {str(e)}",
                "is_solar_panel": True # Assume it was unless proved otherwise
            })
            
    # Save the results
    session_results[session_id] = results
    
    return {"message": "Success", "results": results, "session_id": session_id}

@app.get("/api/results")
async def get_results(session_id: str = Cookie(None)):
    if not session_id or session_id not in session_results:
        return {"results": []}
        
    return {"results": session_results[session_id]}

# Mount frontend directory
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
os.makedirs(frontend_dir, exist_ok=True)
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
