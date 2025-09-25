"""
FastAPI backend for Dataseter Web GUI
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uuid
import os
import json
from pathlib import Path
import tempfile
import shutil

from ..core import DatasetCreator

app = FastAPI(title="Dataseter API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active jobs
jobs = {}

class ExtractionRequest(BaseModel):
    sources: List[Dict[str, Any]]
    chunk_size: int = 512
    overlap: int = 50
    remove_pii: bool = True
    quality_threshold: float = 0.7
    output_format: str = "jsonl"

class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: float
    message: str
    result: Optional[Dict[str, Any]] = None

@app.get("/")
async def root():
    return {"message": "Dataseter API is running"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a file for processing"""
    try:
        # Save uploaded file
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, file.filename)

        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        return {
            "filename": file.filename,
            "path": file_path,
            "size": len(content)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/extract")
async def start_extraction(request: ExtractionRequest, background_tasks: BackgroundTasks):
    """Start dataset extraction job"""
    job_id = str(uuid.uuid4())

    # Initialize job status
    jobs[job_id] = {
        "status": "processing",
        "progress": 0,
        "message": "Starting extraction..."
    }

    # Start background processing
    background_tasks.add_task(process_extraction, job_id, request)

    return {"job_id": job_id}

async def process_extraction(job_id: str, request: ExtractionRequest):
    """Process extraction in background"""
    try:
        creator = DatasetCreator()

        # Add sources
        for source in request.sources:
            source_type = source.get("type")

            if source_type == "pdf":
                creator.add_pdf(source["path"])
            elif source_type == "web":
                creator.add_website(source["url"], source.get("max_depth", 2))
            elif source_type == "office":
                creator.add_office_document(source["path"])
            elif source_type == "ebook":
                creator.add_ebook(source["path"])

        jobs[job_id]["progress"] = 50
        jobs[job_id]["message"] = "Processing documents..."

        # Process dataset
        dataset = creator.process(
            chunk_size=request.chunk_size,
            overlap=request.overlap,
            remove_pii=request.remove_pii,
            quality_threshold=request.quality_threshold
        )

        jobs[job_id]["progress"] = 90
        jobs[job_id]["message"] = "Formatting output..."

        # Save output
        output_dir = tempfile.mkdtemp()
        output_path = os.path.join(output_dir, f"dataset.{request.output_format}")

        if request.output_format == "jsonl":
            dataset.to_jsonl(output_path)
        elif request.output_format == "parquet":
            dataset.to_parquet(output_path)
        elif request.output_format == "csv":
            dataset.to_csv(output_path)

        # Update job status
        jobs[job_id] = {
            "status": "completed",
            "progress": 100,
            "message": "Extraction completed",
            "result": {
                "output_path": output_path,
                "statistics": dataset.statistics,
                "document_count": len(dataset)
            }
        }

    except Exception as e:
        jobs[job_id] = {
            "status": "failed",
            "progress": 0,
            "message": f"Error: {str(e)}"
        }

@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """Get job status"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatus(
        job_id=job_id,
        **jobs[job_id]
    )

@app.get("/download/{job_id}")
async def download_result(job_id: str):
    """Download extraction result"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")

    output_path = job["result"]["output_path"]
    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Output file not found")

    return FileResponse(
        output_path,
        media_type="application/octet-stream",
        filename=os.path.basename(output_path)
    )

@app.delete("/job/{job_id}")
async def delete_job(job_id: str):
    """Delete job and its outputs"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    # Clean up files
    if job.get("result") and job["result"].get("output_path"):
        output_dir = os.path.dirname(job["result"]["output_path"])
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

    del jobs[job_id]

    return {"message": "Job deleted"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "jobs_count": len(jobs)}