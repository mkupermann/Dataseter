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
import logging
from pathlib import Path
import tempfile
import shutil
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import time

from ..core import DatasetCreator

logger = logging.getLogger(__name__)

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
    # Advanced AI features
    chunking_strategy: str = "semantic"
    extract_knowledge: bool = True
    add_metacognitive_annotations: bool = True
    enable_adversarial_testing: bool = True

class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: float
    message: str
    result: Optional[Dict[str, Any]] = None

@app.get("/")
async def root():
    """Serve the frontend HTML"""
    frontend_path = Path(__file__).parent.parent / "web" / "frontend" / "index.html"
    if frontend_path.exists():
        with open(frontend_path, "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    else:
        return {"message": "Dataseter API is running"}

@app.get("/app.js")
async def serve_js():
    """Serve the frontend JavaScript"""
    js_path = Path(__file__).parent.parent / "web" / "frontend" / "app.js"
    if js_path.exists():
        with open(js_path, "r") as f:
            js_content = f.read()
        return HTMLResponse(content=js_content, media_type="application/javascript")
    else:
        raise HTTPException(status_code=404, detail="JavaScript file not found")

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

    # Start processing in a separate thread to avoid blocking
    thread = threading.Thread(target=process_extraction_sync, args=(job_id, request))
    thread.daemon = True
    thread.start()

    return {"job_id": job_id}

def process_extraction_sync(job_id: str, request: ExtractionRequest):
    """Process extraction in background thread"""
    try:
        start_time = time.time()

        # Progress callback to update job status with time estimation
        def update_progress(progress: int, message: str):
            elapsed = time.time() - start_time
            if progress > 0:
                # Estimate total time based on current progress
                estimated_total = elapsed / (progress / 100)
                remaining = estimated_total - elapsed
                if remaining > 60:
                    time_msg = f" (~{int(remaining/60)}m {int(remaining%60)}s remaining)"
                else:
                    time_msg = f" (~{int(remaining)}s remaining)"
                message = message + time_msg if progress < 100 else message

            jobs[job_id]["progress"] = progress
            jobs[job_id]["message"] = message
            logger.info(f"Job {job_id}: {progress}% - {message}")

        creator = DatasetCreator()
        total_sources = len(request.sources)

        # Add sources with progress updates
        for i, source in enumerate(request.sources):
            source_type = source.get("type")
            base_progress = int((i / total_sources) * 30)  # 0-30% for adding sources

            if source_type == "pdf":
                update_progress(base_progress, f"Adding PDF: {source['path'].split('/')[-1]}...")
                creator.add_pdf(source["path"])
                update_progress(base_progress + 5, f"Processed PDF: {source['path'].split('/')[-1]}")
            elif source_type == "web":
                update_progress(base_progress + 5, f"Starting web crawl: {source['url']}...")
                # Add website source (this just registers it, doesn't crawl yet)
                creator.add_website(source["url"], max_depth=source.get("max_depth", 2))
                update_progress(base_progress + 10, f"Initialized crawler for {source['url']}")
            elif source_type == "office":
                update_progress(base_progress, f"Adding Office document: {source['path'].split('/')[-1]}...")
                creator.add_office_document(source["path"])
                update_progress(base_progress + 5, f"Processed Office document: {source['path'].split('/')[-1]}")
            elif source_type == "ebook":
                update_progress(base_progress, f"Adding eBook: {source['path'].split('/')[-1]}...")
                creator.add_ebook(source["path"])
                update_progress(base_progress + 5, f"Processed eBook: {source['path'].split('/')[-1]}")

        update_progress(30, "Extracting content from sources...")

        # Process dataset with progress callback and advanced features
        dataset = creator.process(
            chunk_size=request.chunk_size,
            overlap=request.overlap,
            remove_pii=request.remove_pii,
            quality_threshold=request.quality_threshold,
            chunking_strategy=request.chunking_strategy,
            extract_knowledge=request.extract_knowledge,
            add_metacognitive_annotations=request.add_metacognitive_annotations,
            enable_adversarial_testing=request.enable_adversarial_testing,
            progress_callback=update_progress
        )

        # Check if dataset is empty
        if len(dataset) == 0:
            # Create a minimal dataset with error message
            jobs[job_id] = {
                "status": "completed",
                "progress": 100,
                "message": "Warning: No content extracted",
                "result": {
                    "output_path": None,
                    "statistics": {
                        "total_documents": 0,
                        "total_words": 0,
                        "quality_stats": {"mean": 0},
                        "warning": "No content could be extracted from the sources"
                    },
                    "document_count": 0
                }
            }
            return

        update_progress(90, "Formatting output...")

        # Save output
        output_dir = tempfile.mkdtemp()
        output_path = os.path.join(output_dir, f"dataset.{request.output_format}")

        if request.output_format == "jsonl":
            dataset.to_jsonl(output_path)
        elif request.output_format == "parquet":
            dataset.to_parquet(output_path)
        elif request.output_format == "csv":
            dataset.to_csv(output_path)

        # Get statistics safely
        stats = dataset.statistics if hasattr(dataset, 'statistics') and dataset.statistics else {
            "total_documents": len(dataset),
            "total_words": 0,
            "quality_stats": {"mean": 0}
        }

        # Update job status
        jobs[job_id] = {
            "status": "completed",
            "progress": 100,
            "message": "Extraction completed",
            "result": {
                "output_path": output_path,
                "statistics": stats,
                "document_count": len(dataset)
            }
        }

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Extraction failed for job {job_id}: {error_details}")

        jobs[job_id] = {
            "status": "failed",
            "progress": 0,
            "message": f"Error: {str(e)}",
            "error_details": error_details
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

@app.get("/preview/{job_id}")
async def preview_dataset(job_id: str, limit: int = 5):
    """Preview extraction result with quality metrics"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")

    output_path = job["result"]["output_path"]
    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Output file not found")

    # Read the dataset and analyze quality
    samples = []
    quality_scores = []
    text_lengths = []

    with open(output_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= limit and limit > 0:
                break
            try:
                data = json.loads(line)
                text_len = len(data.get('text', ''))

                # Calculate basic quality score
                quality_score = min(1.0, text_len / 1000)  # Simple length-based score
                if text_len < 50:
                    quality_score *= 0.5  # Penalize very short texts

                samples.append({
                    'id': data.get('id', f'doc_{i}'),
                    'text_preview': data.get('text', '')[:500] + ('...' if len(data.get('text', '')) > 500 else ''),
                    'text_length': text_len,
                    'quality_score': round(quality_score, 2),
                    'has_labels': 'label' in data,
                    'labels': data.get('label', 'No labels')
                })
                quality_scores.append(quality_score)
                text_lengths.append(text_len)
            except json.JSONDecodeError:
                continue

    # Calculate overall metrics
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
    avg_length = sum(text_lengths) / len(text_lengths) if text_lengths else 0

    return {
        'job_id': job_id,
        'total_samples': len(samples),
        'samples': samples,
        'quality_metrics': {
            'average_quality_score': round(avg_quality, 2),
            'average_text_length': round(avg_length, 0),
            'min_text_length': min(text_lengths) if text_lengths else 0,
            'max_text_length': max(text_lengths) if text_lengths else 0,
            'quality_distribution': {
                'excellent': len([s for s in quality_scores if s >= 0.8]),
                'good': len([s for s in quality_scores if 0.6 <= s < 0.8]),
                'fair': len([s for s in quality_scores if 0.4 <= s < 0.6]),
                'poor': len([s for s in quality_scores if s < 0.4])
            }
        },
        'recommendations': generate_recommendations(avg_quality, avg_length, samples)
    }

def generate_recommendations(avg_quality, avg_length, samples):
    """Generate recommendations based on dataset analysis"""
    recommendations = []

    if avg_quality < 0.6:
        recommendations.append("Dataset quality is low. Consider adding more substantial documents.")
    if avg_length < 100:
        recommendations.append("Text chunks are very short. Consider increasing chunk size.")
    if avg_length > 2000:
        recommendations.append("Text chunks are very long. Consider decreasing chunk size for better model training.")

    has_labels = any(s['has_labels'] for s in samples)
    if not has_labels:
        recommendations.append("Dataset has no labels. Add labels for supervised learning tasks.")

    if not recommendations:
        recommendations.append("Dataset quality looks good for training!")

    return recommendations

@app.get("/download/{job_id}")
async def download_result(job_id: str):
    """Download extraction result"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")

    output_path = job["result"].get("output_path")
    if not output_path:
        raise HTTPException(status_code=404, detail="No output file available - extraction may have failed or returned empty results")

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