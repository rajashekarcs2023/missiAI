import os
import argparse
import asyncio
from dotenv import load_dotenv

def setup_environment():
    """
    Set up environment variables for MiniMax MCP API
    """
    # Load environment variables from .env file if it exists
    load_dotenv()
    
    parser = argparse.ArgumentParser(description='MiniMax MCP API Configuration')
    
    parser.add_argument('--api-key', dest='api_key',
                        help='MiniMax API Key')
    
    parser.add_argument('--api-host', dest='api_host',
                        choices=['https://api.minimaxi.chat', 'https://api.minimax.chat'],
                        help='MiniMax API Host (Global: https://api.minimaxi.chat, Mainland: https://api.minimax.chat)')
    
    parser.add_argument('--resource-mode', dest='resource_mode',
                        choices=['url', 'local'],
                        help='Resource mode (url or local)')
    
    parser.add_argument('--base-path', dest='base_path',
                        help='Base path for local resources')
    
    parser.add_argument('--port', dest='port', type=int, default=8005,
                        help='Port to run the API server on')
    
    args = parser.parse_args()
    
    # Set environment variables from arguments if provided
    if args.api_key:
        os.environ['MINIMAX_API_KEY'] = args.api_key
    
    if args.api_host:
        os.environ['MINIMAX_API_HOST'] = args.api_host
    
    if args.resource_mode:
        os.environ['MINIMAX_API_RESOURCE_MODE'] = args.resource_mode
    
    if args.base_path:
        os.environ['MINIMAX_MCP_BASE_PATH'] = args.base_path
    
    return args.port

if __name__ == "__main__":
    port = setup_environment()
    
    # Import and run the API server
    from minimax_mcp_api import app
    import uvicorn
    
    print(f"Starting MiniMax MCP API server on port {port}")
    print(f"API Host: {os.getenv('MINIMAX_API_HOST', 'https://api.minimaxi.chat')}")
    print(f"Resource Mode: {os.getenv('MINIMAX_API_RESOURCE_MODE', 'url')}")
    print(f"Base Path: {os.getenv('MINIMAX_MCP_BASE_PATH', './output')}")
    
    uvicorn.run(app, host="0.0.0.0", port=port)
