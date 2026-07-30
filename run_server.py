import sys
import asyncio
import selectors

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import uvicorn

if __name__ == "__main__":
    config = uvicorn.Config("app.main:app", host="127.0.0.1", port=8000, loop="asyncio")
    server = uvicorn.Server(config)
    if sys.platform == "win32":
        selector = selectors.SelectSelector()
        loop = asyncio.SelectorEventLoop(selector)
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server.serve())
    else:
        server.run()
