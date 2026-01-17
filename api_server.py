#!/usr/bin/env python3
"""
arianna.c API Server
FastAPI backend for web REPL interface

Provides HTTP endpoints for text generation using the arianna.c binary.
Connects the web interface (index.html + chat.js) to the actual C implementation.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import subprocess
import os
import sys
from pathlib import Path

app = FastAPI(
    title="arianna.c API",
    description="REST API for arianna.c AIOS text generation",
    version="1.0.0"
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    """Text generation request"""
    prompt: str = Field(..., min_length=1, max_length=2000, description="Input prompt")
    max_tokens: int = Field(100, ge=10, le=1000, description="Maximum tokens to generate")
    temperature: float = Field(0.8, ge=0.1, le=2.0, description="Sampling temperature")
    mode: str = Field("dynamic", description="Generation mode: 'dynamic' or 'static'")


class GenerateResponse(BaseModel):
    """Text generation response"""
    prompt: str
    generated_text: str
    mode: str
    success: bool
    error: str = None


def find_arianna_binary():
    """Locate the arianna binary"""
    candidates = [
        "./bin/arianna_dynamic",
        "./bin/arianna",
        "./personality/arianna",
    ]
    
    for path in candidates:
        if os.path.exists(path) and os.access(path, os.X_OK):
            return path
    
    return None


def find_weights():
    """Locate the weights file"""
    candidates = [
        "weights/arianna.bin",
        "weights/arianna_core.bin",
    ]
    
    for path in candidates:
        if os.path.exists(path):
            return path
    
    return None


@app.get("/")
async def root():
    """API root - health check"""
    binary = find_arianna_binary()
    weights = find_weights()
    
    return {
        "name": "arianna.c API",
        "version": "1.0.0",
        "status": "online",
        "binary_found": binary is not None,
        "binary_path": binary,
        "weights_found": weights is not None,
        "weights_path": weights,
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    binary = find_arianna_binary()
    weights = find_weights()
    
    if not binary:
        raise HTTPException(
            status_code=503,
            detail="Arianna binary not found. Run 'make dynamic' to build."
        )
    
    if not weights:
        raise HTTPException(
            status_code=503,
            detail="Weights not found. Ensure weights/arianna.bin exists."
        )
    
    return {"status": "healthy", "binary": binary, "weights": weights}


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Generate text using arianna.c
    
    Calls the C binary with the provided prompt and parameters.
    Returns the generated text.
    """
    binary = find_arianna_binary()
    weights = find_weights()
    
    if not binary:
        return GenerateResponse(
            prompt=request.prompt,
            generated_text="",
            mode=request.mode,
            success=False,
            error="Arianna binary not found. Run 'make dynamic' to build."
        )
    
    if not weights:
        return GenerateResponse(
            prompt=request.prompt,
            generated_text="",
            mode=request.mode,
            success=False,
            error="Weights not found. Ensure weights/arianna.bin exists."
        )
    
    try:
        # Build command
        cmd = [
            binary,
            weights,
            request.prompt,
            str(request.max_tokens),
            str(request.temperature)
        ]
        
        # Run generation
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,  # 30 second timeout
            cwd=os.getcwd()
        )
        
        if result.returncode != 0:
            return GenerateResponse(
                prompt=request.prompt,
                generated_text="",
                mode=request.mode,
                success=False,
                error=f"Generation failed: {result.stderr}"
            )
        
        # Extract generated text from output
        output = result.stdout.strip()
        
        return GenerateResponse(
            prompt=request.prompt,
            generated_text=output,
            mode=request.mode,
            success=True
        )
        
    except subprocess.TimeoutExpired:
        return GenerateResponse(
            prompt=request.prompt,
            generated_text="",
            mode=request.mode,
            success=False,
            error="Generation timed out (30s limit)"
        )
    except Exception as e:
        return GenerateResponse(
            prompt=request.prompt,
            generated_text="",
            mode=request.mode,
            success=False,
            error=f"Unexpected error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    
    print("Starting arianna.c API server...")
    print("API docs available at: http://localhost:8000/docs")
    print("Web interface: Open index.html in a browser")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
