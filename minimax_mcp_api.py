import os
import json
import uuid
import time
import base64
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Header, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("minimax-mcp")

# Initialize FastAPI app
app = FastAPI(title="MiniMax MCP API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables
API_KEY = os.getenv("MINIMAX_API_KEY")
API_HOST = os.getenv("MINIMAX_API_HOST", "https://api.minimaxi.chat")
RESOURCE_MODE = os.getenv("MINIMAX_API_RESOURCE_MODE", "url")
BASE_PATH = os.getenv("MINIMAX_MCP_BASE_PATH", "./output")

# Create base directory for local resources if it doesn't exist
if RESOURCE_MODE == "local":
    os.makedirs(BASE_PATH, exist_ok=True)

# Task storage
tasks = {}

# MCP Protocol Models
class Resource(BaseModel):
    uri: str
    type: str

class MCPListResourcesRequest(BaseModel):
    cursor: Optional[str] = None

class MCPListResourcesResponse(BaseModel):
    resources: List[Resource]
    cursor: Optional[str] = None

class MCPReadResourceRequest(BaseModel):
    uri: str

class MCPReadResourceResponse(BaseModel):
    content: str

# MiniMax API Models
class TextToAudioRequest(BaseModel):
    text: str
    voice_id: str = "default"
    speed: float = 1.0
    format: str = "mp3"

class VoiceCloneRequest(BaseModel):
    name: str
    audio_files: List[str]
    description: Optional[str] = None

class GenerateVideoRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    width: int = 512
    height: int = 512
    duration: int = 3
    fps: int = 24
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
    async_mode: bool = True

class TextToImageRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    width: int = 512
    height: int = 512
    guidance_scale: float = 7.5
    num_inference_steps: int = 50

class QueryVideoGenerationRequest(BaseModel):
    task_id: str

# Helper functions
async def call_minimax_api(endpoint: str, data: Dict[str, Any], method: str = "POST") -> Dict[str, Any]:
    """
    Call the MiniMax API with the given endpoint and data
    """
    if not API_KEY:
        raise HTTPException(status_code=500, detail="MINIMAX_API_KEY not set")
    
    url = f"{API_HOST}{endpoint}"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    async with httpx.AsyncClient() as client:
        try:
            if method == "POST":
                response = await client.post(url, json=data, headers=headers)
            elif method == "GET":
                response = await client.get(url, headers=headers)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            if response.status_code != 200:
                logger.error(f"MiniMax API error: {response.status_code} {response.text}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"MiniMax API error: {response.text}"
                )
            
            return response.json()
        except httpx.RequestError as e:
            logger.error(f"Error calling MiniMax API: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error calling MiniMax API: {str(e)}")

def save_resource_locally(content: bytes, extension: str) -> str:
    """
    Save a resource locally and return the path
    """
    filename = f"{uuid.uuid4()}.{extension}"
    filepath = os.path.join(BASE_PATH, filename)
    
    with open(filepath, "wb") as f:
        f.write(content)
    
    return filepath

# MCP Protocol Endpoints
@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/list_resources")
async def list_resources(request: MCPListResourcesRequest) -> MCPListResourcesResponse:
    """
    List available resources (voices, models, etc.)
    """
    # For now, we'll return a fixed set of resources
    voices = [
        Resource(uri="voice://default", type="voice"),
        Resource(uri="voice://male", type="voice"),
        Resource(uri="voice://female", type="voice"),
        Resource(uri="model://text-to-image", type="model"),
        Resource(uri="model://text-to-video", type="model"),
    ]
    
    return MCPListResourcesResponse(resources=voices)

@app.post("/read_resource")
async def read_resource(request: MCPReadResourceRequest) -> MCPReadResourceResponse:
    """
    Read a resource's content
    """
    # For now, we'll return fixed content for known resources
    if request.uri.startswith("voice://"):
        voice_id = request.uri.replace("voice://", "")
        return MCPReadResourceResponse(content=json.dumps({
            "id": voice_id,
            "name": voice_id.capitalize(),
            "description": f"A {voice_id} voice"
        }))
    elif request.uri.startswith("model://"):
        model_id = request.uri.replace("model://", "")
        return MCPReadResourceResponse(content=json.dumps({
            "id": model_id,
            "name": model_id.capitalize(),
            "description": f"A {model_id} model"
        }))
    else:
        raise HTTPException(status_code=404, detail=f"Resource not found: {request.uri}")

# MiniMax API Endpoints
@app.post("/text_to_audio")
async def text_to_audio(request: TextToAudioRequest, x_api_key: Optional[str] = Header(None)):
    """
    Convert text to audio using MiniMax TTS
    """
    if x_api_key and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    try:
        # Call MiniMax API
        response = await call_minimax_api(
            "/text_to_speech/v1/generate",
            {
                "text": request.text,
                "voice_id": request.voice_id,
                "speed": request.speed,
                "format": request.format
            }
        )
        
        # Handle response based on resource mode
        if RESOURCE_MODE == "url":
            return {"audio_url": response.get("audio_url")}
        else:
            # Download audio and save locally
            async with httpx.AsyncClient() as client:
                audio_response = await client.get(response.get("audio_url"))
                if audio_response.status_code != 200:
                    raise HTTPException(status_code=500, detail="Failed to download audio")
                
                filepath = save_resource_locally(audio_response.content, request.format)
                return {"audio_path": filepath}
    except Exception as e:
        logger.error(f"Error in text_to_audio: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list_voices")
async def list_voices(x_api_key: Optional[str] = Header(None)):
    """
    List available voices
    """
    if x_api_key and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    try:
        # Call MiniMax API
        response = await call_minimax_api(
            "/text_to_speech/v1/voices",
            {},
            method="GET"
        )
        
        return {"voices": response.get("voices", [])}
    except Exception as e:
        logger.error(f"Error in list_voices: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/voice_clone")
async def voice_clone(request: VoiceCloneRequest, x_api_key: Optional[str] = Header(None)):
    """
    Clone a voice from audio samples
    """
    if x_api_key and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    try:
        # Call MiniMax API
        response = await call_minimax_api(
            "/text_to_speech/v1/voice_clone",
            {
                "name": request.name,
                "audio_files": request.audio_files,
                "description": request.description
            }
        )
        
        return {"voice_id": response.get("voice_id")}
    except Exception as e:
        logger.error(f"Error in voice_clone: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_video")
async def generate_video(request: GenerateVideoRequest, x_api_key: Optional[str] = Header(None)):
    """
    Generate a video from a text prompt
    """
    if x_api_key and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    try:
        # Call MiniMax API
        response = await call_minimax_api(
            "/text_to_video/v1/generate",
            {
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt,
                "width": request.width,
                "height": request.height,
                "duration": request.duration,
                "fps": request.fps,
                "guidance_scale": request.guidance_scale,
                "num_inference_steps": request.num_inference_steps,
                "async": request.async_mode
            }
        )
        
        task_id = response.get("task_id")
        tasks[task_id] = {
            "status": "processing",
            "created_at": time.time(),
            "type": "video",
            "request": request.dict()
        }
        
        return {"task_id": task_id}
    except Exception as e:
        logger.error(f"Error in generate_video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/text_to_image")
async def text_to_image(request: TextToImageRequest, x_api_key: Optional[str] = Header(None)):
    """
    Generate an image from a text prompt
    """
    if x_api_key and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    try:
        # Call MiniMax API
        response = await call_minimax_api(
            "/text_to_image/v1/generate",
            {
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt,
                "width": request.width,
                "height": request.height,
                "guidance_scale": request.guidance_scale,
                "num_inference_steps": request.num_inference_steps
            }
        )
        
        # Handle response based on resource mode
        if RESOURCE_MODE == "url":
            return {"image_url": response.get("image_url")}
        else:
            # Download image and save locally
            async with httpx.AsyncClient() as client:
                image_response = await client.get(response.get("image_url"))
                if image_response.status_code != 200:
                    raise HTTPException(status_code=500, detail="Failed to download image")
                
                filepath = save_resource_locally(image_response.content, "png")
                return {"image_path": filepath}
    except Exception as e:
        logger.error(f"Error in text_to_image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query_video_generation")
async def query_video_generation(request: QueryVideoGenerationRequest, x_api_key: Optional[str] = Header(None)):
    """
    Query the status of a video generation task
    """
    if x_api_key and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    try:
        # Call MiniMax API
        response = await call_minimax_api(
            "/text_to_video/v1/query",
            {"task_id": request.task_id}
        )
        
        status = response.get("status")
        if status == "succeeded":
            video_url = response.get("video_url")
            
            # Handle response based on resource mode
            if RESOURCE_MODE == "url":
                return {"status": status, "video_url": video_url}
            else:
                # Download video and save locally
                async with httpx.AsyncClient() as client:
                    video_response = await client.get(video_url)
                    if video_response.status_code != 200:
                        raise HTTPException(status_code=500, detail="Failed to download video")
                    
                    filepath = save_resource_locally(video_response.content, "mp4")
                    return {"status": status, "video_path": filepath}
        
        return {"status": status}
    except Exception as e:
        logger.error(f"Error in query_video_generation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stream/{task_id}")
async def stream_task_updates(task_id: str, x_api_key: Optional[str] = Header(None)):
    """
    Stream updates for a task
    """
    if x_api_key and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    
    async def event_stream():
        task = tasks[task_id]
        yield f"data: {json.dumps({'status': task['status']})}\n\n"
        
        # If task is video generation, poll for updates
        if task["type"] == "video":
            while task["status"] == "processing":
                try:
                    response = await call_minimax_api(
                        "/text_to_video/v1/query",
                        {"task_id": task_id}
                    )
                    
                    status = response.get("status")
                    task["status"] = status
                    
                    yield f"data: {json.dumps({'status': status})}\n\n"
                    
                    if status != "processing":
                        if status == "succeeded":
                            video_url = response.get("video_url")
                            if RESOURCE_MODE == "url":
                                yield f"data: {json.dumps({'status': status, 'video_url': video_url})}\n\n"
                            else:
                                # Download video and save locally
                                async with httpx.AsyncClient() as client:
                                    video_response = await client.get(video_url)
                                    if video_response.status_code == 200:
                                        filepath = save_resource_locally(video_response.content, "mp4")
                                        yield f"data: {json.dumps({'status': status, 'video_path': filepath})}\n\n"
                        break
                    
                    await asyncio.sleep(2)
                except Exception as e:
                    logger.error(f"Error polling task {task_id}: {str(e)}")
                    yield f"data: {json.dumps({'status': 'error', 'error': str(e)})}\n\n"
                    break
    
    return StreamingResponse(event_stream(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    import asyncio
    
    port = int(os.getenv("PORT", "8005"))
    uvicorn.run(app, host="0.0.0.0", port=port)
