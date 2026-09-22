"""
Least-outstanding-requests streaming proxy for data-parallel vLLM replicas (one per B70).
Usage: python3 scripts/lb.py --port 8000 --backends http://localhost:8100,http://localhost:8101
"""
import argparse
import asyncio
import aiohttp
from aiohttp import web

HOP = {"host", "content-length", "transfer-encoding", "connection"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--backends", required=True)
    args = ap.parse_args()
    backends = args.backends.split(",")
    inflight = {b: 0 for b in backends}
    session: dict = {}

    async def on_startup(app):
        session["s"] = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None),
                                             connector=aiohttp.TCPConnector(limit=0))

    async def on_cleanup(app):
        await session["s"].close()

    async def proxy(request: web.Request):
        b = min(backends, key=lambda x: inflight[x])
        inflight[b] += 1
        try:
            body = await request.read()
            headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP}
            async with session["s"].request(request.method, b + request.rel_url.path_qs,
                                            data=body, headers=headers) as r:
                resp = web.StreamResponse(status=r.status,
                                          headers={k: v for k, v in r.headers.items() if k.lower() not in HOP})
                await resp.prepare(request)
                async for chunk in r.content.iter_any():
                    await resp.write(chunk)
                await resp.write_eof()
                return resp
        finally:
            inflight[b] -= 1

    app = web.Application(client_max_size=1 << 30)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.router.add_route("*", "/{tail:.*}", proxy)
    web.run_app(app, port=args.port, access_log=None)


if __name__ == "__main__":
    main()
