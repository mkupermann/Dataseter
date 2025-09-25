#!/usr/bin/env python3
"""
Minimal server startup script for Dataseter Web GUI
"""
import sys
import subprocess
import os

# Install minimal dependencies
print("Installing minimal dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                "fastapi", "uvicorn", "pydantic", "python-multipart",
                "pyyaml", "tqdm", "html2text", "click", "rich"])

# Start the server
print("\nStarting Dataseter Web Server...")
print("Server will be available at: http://localhost:8080")
print("Press Ctrl+C to stop\n")

os.system(f"{sys.executable} -m uvicorn src.api.main:app --host 0.0.0.0 --port 8080 --reload")