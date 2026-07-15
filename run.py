import sys
import asyncio
import uvicorn
from app.main import app

if __name__ == "__main__":
    if sys.platform == "win32":
        # Force Windows SelectorEventLoop policy and instance
        asyncio.WindowsProactorEventLoopPolicy = asyncio.WindowsSelectorEventLoopPolicy
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        loop = asyncio.SelectorEventLoop()
        asyncio.set_event_loop(loop)
    else:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
    config = uvicorn.Config(app, host="127.0.0.1", port=8000, loop="asyncio")
    server = uvicorn.Server(config)
    loop.run_until_complete(server.serve())
