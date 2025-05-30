import os
import json
import logging
import asyncio
import time
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("minimax-direct-api")

# Initialize FastAPI app
app = FastAPI(title="MiniMax Direct API", version="1.0.0")

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

# Pydantic models
class DirectRequest(BaseModel):
    tool: str
    parameters: Dict[str, Any]
    user_id: Optional[str] = None

class DirectResponse(BaseModel):
    result: Any
    success: bool
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

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

# API endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Get version information
        import platform
        import sys
        
        # Check for required environment variables
        required_vars = [
            "MINIMAX_API_KEY",
            "MINIMAX_API_HOST"
        ]
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        return {
            "status": "healthy",
            "server_running": True,
            "missing_env_vars": missing_vars if missing_vars else None,
            "timestamp": str(asyncio.get_event_loop().time()),
            "python_version": sys.version,
            "platform": platform.platform(),
            "api_version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Error in health check endpoint: {str(e)}")
        return {"status": "error", "error": str(e)}

@app.post("/direct")
async def direct_api(request: DirectRequest):
    """
    Direct API endpoint for MiniMax tools
    """
    try:
        tool = request.tool
        parameters = request.parameters
        
        # Log the request
        logger.info(f"Direct API request: tool={tool}, parameters={parameters}")
        
        # Check for duplicate requests (simple in-memory cache of recent requests)
        if not hasattr(app.state, "recent_requests"):
            app.state.recent_requests = {}
            
        request_hash = hash(f"{tool}_{json.dumps(parameters)}")
        current_time = time.time()
        
        # Check if this is a duplicate request within the last 30 seconds
        if request_hash in app.state.recent_requests:
            last_time, last_response = app.state.recent_requests[request_hash]
            if current_time - last_time < 30:  # 30 second window for duplicates
                logger.info("Duplicate request detected within 30 seconds, returning cached response")
                return JSONResponse(content=last_response)
        
        # Execute the requested tool
        result = None
        metadata = {}
        
        if tool == "text_to_audio":
            # Extract parameters
            text = parameters.get("text")
            voice_id = parameters.get("voice_id", "Grinch")
            speed = parameters.get("speed", 1.0)
            format_type = parameters.get("format", "mp3")
            model = parameters.get("model", "speech-02-hd")
            
            # Validate parameters
            if not text:
                raise ValueError("Missing required parameter: text")
            
            # Get the group ID from the API key
            # MiniMax API keys contain the group ID information
            group_id = "1928493199890846380"  # Hardcoded from the API key JWT
            logger.info(f"Using group ID: {group_id}")
            
            # Call MiniMax API with T2A v2 endpoint
            url = f"{API_HOST}/v1/t2a_v2?GroupId={group_id}"
            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": model,
                "text": text,
                "stream": False,
                "voice_setting": {
                    "voice_id": voice_id,
                    "speed": speed,
                    "vol": 1,
                    "pitch": 0
                },
                "audio_setting": {
                    "sample_rate": 32000,
                    "bitrate": 128000,
                    "format": format_type,
                    "channel": 1
                }
            }
            
            logger.info(f"Calling MiniMax T2A v2 API with data: {data}")
            
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=data, headers=headers)
                
                if response.status_code != 200:
                    logger.error(f"MiniMax API error: {response.status_code} {response.text}")
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"MiniMax API error: {response.text}"
                    )
                
                response_data = response.json()
                logger.info(f"MiniMax API response: {response_data}")
                
                # Extract audio data
                if "data" in response_data and "audio" in response_data["data"]:
                    audio_hex = response_data["data"]["audio"]
                    audio_bytes = bytes.fromhex(audio_hex)
                    
                    # Handle response based on resource mode
                    if RESOURCE_MODE == "url":
                        # Save audio to a file and return the URL
                        filepath = save_resource_locally(audio_bytes, format_type)
                        result = {"audio_path": filepath}
                        
                        # If subtitle file is available, include it
                        if "subtitle_file" in response_data["data"]:
                            result["subtitle_url"] = response_data["data"]["subtitle_file"]
                    else:
                        # Save audio to a file and return the path
                        filepath = save_resource_locally(audio_bytes, format_type)
                        result = {"audio_path": filepath}
                    
                    # Extract metadata
                    metadata = {}
                    if "extra_info" in response_data:
                        metadata = response_data["extra_info"]
                        metadata["voice_id"] = voice_id
                else:
                    result = {"error": "No audio data in response"}
                    metadata = {"response": response_data}
        
        elif tool == "list_voices":
            # Call MiniMax API
            response = await call_minimax_api(
                "/v1/text_to_speech/voices",
                {},
                method="GET"
            )
            
            result = {"voices": response.get("voices", [])}
        
        elif tool == "voice_clone":
            # Extract parameters
            name = parameters.get("name")
            audio_files = parameters.get("audio_files")
            description = parameters.get("description")
            
            # Validate parameters
            if not name:
                raise ValueError("Missing required parameter: name")
            if not audio_files or not isinstance(audio_files, list):
                raise ValueError("Missing required parameter: audio_files (must be a list)")
            
            # Call MiniMax API
            response = await call_minimax_api(
                "/v1/text_to_speech/voice_clone",
                {
                    "name": name,
                    "audio_files": audio_files,
                    "description": description
                }
            )
            
            result = {"voice_id": response.get("voice_id")}
        
        elif tool == "generate_video":
            # Extract parameters
            prompt = parameters.get("prompt")
            negative_prompt = parameters.get("negative_prompt")
            width = parameters.get("width", 512)
            height = parameters.get("height", 512)
            duration = parameters.get("duration", 3)
            fps = parameters.get("fps", 24)
            guidance_scale = parameters.get("guidance_scale", 7.5)
            num_inference_steps = parameters.get("num_inference_steps", 50)
            async_mode = parameters.get("async_mode", True)
            
            # Validate parameters
            if not prompt:
                raise ValueError("Missing required parameter: prompt")
            
            # Call MiniMax API
            response = await call_minimax_api(
                "/v1/text_to_video",
                {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "width": width,
                    "height": height,
                    "duration": duration,
                    "fps": fps,
                    "guidance_scale": guidance_scale,
                    "num_inference_steps": num_inference_steps,
                    "async": async_mode
                }
            )
            
            task_id = response.get("task_id")
            tasks[task_id] = {
                "status": "processing",
                "created_at": time.time(),
                "type": "video",
                "request": parameters
            }
            
            result = {"task_id": task_id}
            
            # If not async, poll for completion
            if not async_mode:
                max_retries = 60  # 5 minutes (5s * 60)
                for i in range(max_retries):
                    await asyncio.sleep(5)  # Wait 5 seconds between polls
                    
                    try:
                        poll_response = await call_minimax_api(
                            "/v1/text_to_video/query",
                            {"task_id": task_id}
                        )
                        
                        status = poll_response.get("status")
                        tasks[task_id]["status"] = status
                        
                        if status != "processing":
                            if status == "succeeded":
                                video_url = poll_response.get("video_url")
                                if RESOURCE_MODE == "url":
                                    result = {"task_id": task_id, "status": status, "video_url": video_url}
                                else:
                                    # Download video and save locally
                                    async with httpx.AsyncClient() as client:
                                        video_response = await client.get(video_url)
                                        if video_response.status_code == 200:
                                            filepath = save_resource_locally(video_response.content, "mp4")
                                            result = {"task_id": task_id, "status": status, "video_path": filepath}
                            else:
                                result = {"task_id": task_id, "status": status}
                            break
                    except Exception as e:
                        logger.error(f"Error polling task {task_id}: {str(e)}")
                        break
        
        elif tool == "text_to_image":
            # Extract parameters
            prompt = parameters.get("prompt")
            negative_prompt = parameters.get("negative_prompt")
            width = parameters.get("width", 512)
            height = parameters.get("height", 512)
            guidance_scale = parameters.get("guidance_scale", 7.5)
            num_inference_steps = parameters.get("num_inference_steps", 50)
            
            # Validate parameters
            if not prompt:
                raise ValueError("Missing required parameter: prompt")
            
            # Call MiniMax API
            response = await call_minimax_api(
                "/v1/text_to_image",
                {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "width": width,
                    "height": height,
                    "guidance_scale": guidance_scale,
                    "num_inference_steps": num_inference_steps
                }
            )
            
            # Handle response based on resource mode
            if RESOURCE_MODE == "url":
                result = {"image_url": response.get("image_url")}
            else:
                # Download image and save locally
                async with httpx.AsyncClient() as client:
                    image_response = await client.get(response.get("image_url"))
                    if image_response.status_code != 200:
                        raise HTTPException(status_code=500, detail="Failed to download image")
                    
                    filepath = save_resource_locally(image_response.content, "png")
                    result = {"image_path": filepath}
        
        elif tool == "query_video_generation":
            # Extract parameters
            task_id = parameters.get("task_id")
            
            # Validate parameters
            if not task_id:
                raise ValueError("Missing required parameter: task_id")
            
            # Call MiniMax API
            response = await call_minimax_api(
                "/v1/text_to_video/query",
                {"task_id": task_id}
            )
            
            status = response.get("status")
            if status == "succeeded":
                video_url = response.get("video_url")
                
                # Handle response based on resource mode
                if RESOURCE_MODE == "url":
                    result = {"status": status, "video_url": video_url}
                else:
                    # Download video and save locally
                    async with httpx.AsyncClient() as client:
                        video_response = await client.get(video_url)
                        if video_response.status_code != 200:
                            raise HTTPException(status_code=500, detail="Failed to download video")
                        
                        filepath = save_resource_locally(video_response.content, "mp4")
                        result = {"status": status, "video_path": filepath}
            else:
                result = {"status": status}
        
        else:
            raise ValueError(f"Unsupported tool: {tool}")
        
        # Prepare the response
        response_json = {
            "result": result,
            "success": True,
            "metadata": metadata
        }
        
        # Cache the successful response for duplicate detection
        app.state.recent_requests[request_hash] = (time.time(), response_json)
        
        return JSONResponse(content=response_json)
    
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Error processing direct API request: {str(e)}")
        logger.error(error_traceback)
        
        return JSONResponse(
            status_code=200,  # Return 200 for compatibility with frontend
            content={
                "result": None,
                "success": False,
                "error": str(e),
                "metadata": {
                    "error_type": type(e).__name__,
                    "traceback": error_traceback
                }
            }
        )

@app.get("/stream/{task_id}")
async def stream_task_updates(task_id: str):
    """
    Stream updates for a task
    """
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
                        "/v1/text_to_video/query",
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

@app.post("/debug")
async def debug_endpoint(request: Request):
    """Debug endpoint to log request details"""
    try:
        # Log the raw request details
        raw_body = await request.body()
        logger.info(f"Debug endpoint received raw request: {raw_body}")
        logger.info(f"Request headers: {request.headers}")
        
        # Log the request to a file for inspection
        with open("minimax_debug.log", "a") as f:
            f.write(f"\n\n--- NEW REQUEST {datetime.now().isoformat()} ---\n")
            f.write(f"Raw body: {raw_body}\n")
            f.write(f"Headers: {request.headers}\n")
        
        # Get the JSON body from the request
        try:
            body = await request.json()
            with open("minimax_debug.log", "a") as f:
                f.write(f"Parsed JSON: {body}\n")
            
            return JSONResponse(content={"status": "ok", "received": body})
        except Exception as e:
            logger.error(f"Error parsing JSON: {str(e)}")
            with open("minimax_debug.log", "a") as f:
                f.write(f"Error parsing JSON: {str(e)}\n")
            
            return JSONResponse(
                status_code=400,
                content={"error": f"Error parsing JSON: {str(e)}", "raw_body": raw_body.decode('utf-8', errors='replace')}
            )
    except Exception as e:
        logger.error(f"Error in debug endpoint: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error in debug endpoint: {str(e)}"}
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8005"))
    uvicorn.run(app, host="0.0.0.0", port=port)
