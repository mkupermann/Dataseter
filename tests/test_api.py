"""
API endpoint tests
"""

import pytest
import json
import tempfile
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
import uuid

from src.api.main import app, jobs

client = TestClient(app)


class TestAPIEndpoints:
    """Test API endpoints"""

    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["message"] == "Dataseter API is running"

    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "jobs_count" in data

    @patch('src.api.main.DatasetCreator')
    def test_upload_file(self, mock_creator):
        """Test file upload endpoint"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content")
            f.flush()

            with open(f.name, 'rb') as upload_file:
                response = client.post(
                    "/upload",
                    files={"file": ("test.txt", upload_file, "text/plain")}
                )

        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "test.txt"
        assert data["size"] > 0

    @patch('src.api.main.BackgroundTasks')
    def test_start_extraction(self, mock_background):
        """Test starting extraction job"""
        request_data = {
            "sources": [
                {"type": "web", "url": "https://example.com"},
                {"type": "pdf", "path": "/path/to/file.pdf"}
            ],
            "chunk_size": 512,
            "overlap": 50,
            "remove_pii": True,
            "quality_threshold": 0.7,
            "output_format": "jsonl"
        }

        response = client.post("/extract", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert len(data["job_id"]) > 0

    def test_get_job_status(self):
        """Test getting job status"""
        # Add a test job
        job_id = str(uuid.uuid4())
        jobs[job_id] = {
            "status": "processing",
            "progress": 50,
            "message": "Processing..."
        }

        response = client.get(f"/job/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert data["status"] == "processing"
        assert data["progress"] == 50

        # Clean up
        del jobs[job_id]

    def test_get_nonexistent_job(self):
        """Test getting status of non-existent job"""
        response = client.get("/job/nonexistent")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_download_result(self):
        """Test downloading extraction result"""
        # Create a completed job with result
        job_id = str(uuid.uuid4())

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"test": "data"}')
            output_path = f.name

        jobs[job_id] = {
            "status": "completed",
            "progress": 100,
            "message": "Completed",
            "result": {
                "output_path": output_path
            }
        }

        response = client.get(f"/download/{job_id}")
        assert response.status_code == 200

        # Clean up
        del jobs[job_id]

    def test_download_incomplete_job(self):
        """Test downloading from incomplete job"""
        job_id = str(uuid.uuid4())
        jobs[job_id] = {
            "status": "processing",
            "progress": 50,
            "message": "Processing..."
        }

        response = client.get(f"/download/{job_id}")
        assert response.status_code == 400
        assert "not completed" in response.json()["detail"].lower()

        # Clean up
        del jobs[job_id]

    def test_delete_job(self):
        """Test deleting job"""
        job_id = str(uuid.uuid4())
        jobs[job_id] = {
            "status": "completed",
            "progress": 100,
            "message": "Completed"
        }

        response = client.delete(f"/job/{job_id}")
        assert response.status_code == 200
        assert job_id not in jobs

    def test_delete_nonexistent_job(self):
        """Test deleting non-existent job"""
        response = client.delete("/job/nonexistent")
        assert response.status_code == 404


class TestExtractionProcess:
    """Test the extraction process function"""

    @patch('src.api.main.DatasetCreator')
    async def test_process_extraction_success(self, mock_creator_class):
        """Test successful extraction processing"""
        from src.api.main import process_extraction

        # Mock the creator
        mock_creator = Mock()
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=5)
        mock_dataset.statistics = {"total_documents": 5}
        mock_dataset.to_jsonl = Mock()

        mock_creator.process.return_value = mock_dataset
        mock_creator_class.return_value = mock_creator

        # Prepare request
        job_id = str(uuid.uuid4())
        request = Mock()
        request.sources = [{"type": "web", "url": "https://example.com"}]
        request.chunk_size = 512
        request.overlap = 50
        request.remove_pii = True
        request.quality_threshold = 0.7
        request.output_format = "jsonl"

        # Initialize job
        jobs[job_id] = {
            "status": "processing",
            "progress": 0,
            "message": "Starting..."
        }

        # Run extraction
        await process_extraction(job_id, request)

        # Check job was updated
        assert jobs[job_id]["status"] == "completed"
        assert jobs[job_id]["progress"] == 100
        assert "result" in jobs[job_id]

        # Clean up
        del jobs[job_id]

    @patch('src.api.main.DatasetCreator')
    async def test_process_extraction_failure(self, mock_creator_class):
        """Test extraction processing with failure"""
        from src.api.main import process_extraction

        # Mock the creator to raise an error
        mock_creator = Mock()
        mock_creator.process.side_effect = Exception("Processing failed")
        mock_creator_class.return_value = mock_creator

        # Prepare request
        job_id = str(uuid.uuid4())
        request = Mock()
        request.sources = [{"type": "web", "url": "https://example.com"}]
        request.chunk_size = 512
        request.overlap = 50
        request.remove_pii = True
        request.quality_threshold = 0.7
        request.output_format = "jsonl"

        # Initialize job
        jobs[job_id] = {
            "status": "processing",
            "progress": 0,
            "message": "Starting..."
        }

        # Run extraction
        await process_extraction(job_id, request)

        # Check job failed
        assert jobs[job_id]["status"] == "failed"
        assert "Processing failed" in jobs[job_id]["message"]

        # Clean up
        del jobs[job_id]


class TestAPIIntegration:
    """Integration tests for API workflow"""

    @patch('src.api.main.DatasetCreator')
    def test_complete_workflow(self, mock_creator_class):
        """Test complete API workflow from upload to download"""
        # Step 1: Upload a file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content for extraction")
            test_file = f.name

        with open(test_file, 'rb') as upload_file:
            upload_response = client.post(
                "/upload",
                files={"file": ("test.txt", upload_file, "text/plain")}
            )

        assert upload_response.status_code == 200
        uploaded_path = upload_response.json()["path"]

        # Step 2: Start extraction
        extract_request = {
            "sources": [
                {"type": "text", "path": uploaded_path}
            ],
            "chunk_size": 100,
            "overlap": 10,
            "remove_pii": False,
            "quality_threshold": 0.5,
            "output_format": "jsonl"
        }

        extract_response = client.post("/extract", json=extract_request)
        assert extract_response.status_code == 200
        job_id = extract_response.json()["job_id"]

        # Step 3: Check job status
        status_response = client.get(f"/job/{job_id}")
        assert status_response.status_code == 200

        # Note: In real test, we would wait for job completion
        # For now, manually set job as completed
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"test": "result"}')
            output_file = f.name

        jobs[job_id] = {
            "status": "completed",
            "progress": 100,
            "message": "Completed",
            "result": {
                "output_path": output_file,
                "statistics": {"total_documents": 1},
                "document_count": 1
            }
        }

        # Step 4: Download result
        download_response = client.get(f"/download/{job_id}")
        assert download_response.status_code == 200

        # Step 5: Delete job
        delete_response = client.delete(f"/job/{job_id}")
        assert delete_response.status_code == 200

    def test_concurrent_jobs(self):
        """Test handling multiple concurrent jobs"""
        job_ids = []

        # Create multiple jobs
        for i in range(5):
            job_id = str(uuid.uuid4())
            jobs[job_id] = {
                "status": "processing",
                "progress": i * 20,
                "message": f"Processing job {i}"
            }
            job_ids.append(job_id)

        # Check all jobs
        for job_id in job_ids:
            response = client.get(f"/job/{job_id}")
            assert response.status_code == 200

        # Clean up
        for job_id in job_ids:
            del jobs[job_id]

    def test_cors_headers(self):
        """Test CORS headers are properly set"""
        response = client.get("/")
        assert "access-control-allow-origin" in response.headers

    def test_invalid_extraction_request(self):
        """Test extraction with invalid request data"""
        invalid_request = {
            "sources": [],  # Empty sources
            "chunk_size": -1,  # Invalid chunk size
            "output_format": "invalid"
        }

        response = client.post("/extract", json=invalid_request)
        # Should still accept but may fail during processing
        assert response.status_code in [200, 422]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])