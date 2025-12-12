# project/server/mcp_server.py
import asyncio
from mcp_tool.MCPToolServer import MCPToolServer

async def main():
    server = MCPToolServer(name="fashion-reco-mcp")
    await server.run()

if __name__ == "__main__":
    asyncio.run(main())
