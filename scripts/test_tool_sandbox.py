import asyncio
import sys
sys.path.insert(0, "/home/c00954327/slime-ascend/slime/examples/retool")
from tool_sandbox import tool_registry

async def main():
    r1 = await tool_registry.execute_tool("code_interpreter", {"code": "print(6*7)"})
    print("CASE1 normal compute:", repr(r1[:120]))
    r2 = await tool_registry.execute_tool("code_interpreter", {"code": "import os\nprint(os.listdir('/'))"})
    print("CASE2 import os:", repr(r2[:200]))
    r3 = await tool_registry.execute_tool("code_interpreter", {"code": "while True: pass"})
    print("CASE3 infinite loop:", repr(r3[:200]))

asyncio.run(main())
