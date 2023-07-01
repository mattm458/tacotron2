from aiohttp import web


async def index(request):
    return web.Response(text="Hello, world")


def do_server(port: int):
    app = web.Application()
    app.add_routes([web.get("/", index)])

    web.run_app(app)
