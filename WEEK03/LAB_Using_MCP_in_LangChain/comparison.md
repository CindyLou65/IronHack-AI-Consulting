# MCP Integration vs Direct API Integration (LangChain Lab)

Had many issues with environment setup and compatability issues, with many workarounds to resolve.

## What MCP adds
MCP (Model Context Protocol) standardizes how external capabilities are exposed to an LLM:
- **Tools**: callable actions (functions)
- **Resources**: read-only context (documents/config/data)
- A consistent protocol/transport (HTTP/stdio) decoupling agent logic from tool implementation.

## MCP + LangChain (what I built)
- MCP server exposes tools `add`, `multiply` and resource `info://rules`.
- LangChain loads tools via `MultiServerMCPClient.get_tools()` (returned as `StructuredTool`).
- Resources were listed via adapter; resource reading was done via MCP Python SDK due to adapter version limitations.
- Agent created with `create_agent(...)` uses the MCP tools to solve practical tasks.

## Direct API integration (without MCP)
Direct integration typically means:
- You call APIs yourself (requests/SDK)
- You manually wrap functions as LangChain tools
- You handle auth, schemas, retries, and resource/document retrieval logic inside your application

## When to use MCP
Use MCP when:
- You want a **standard interface** to tools/resources across different providers
- Multiple agents/apps should reuse the same tool server
- You want to swap tool implementations without changing the agent code
- You want centralized governance (logging, access control) around tool execution

## When to use direct API integration
Use direct integration when:
- You only need 1–2 APIs and want minimal moving parts
- Performance/latency is critical and you don’t want another hop
- You need custom request logic not easily expressed as MCP tools/resources

## Trade-offs
### Pros of MCP
- Standard protocol, reusable across projects
- Clean separation: agent code vs capability implementation
- Easier multi-tool/multi-server orchestration

### Cons of MCP
- Extra component to run/maintain (server)
- Version mismatches across libraries can affect helper methods
- Debugging involves tool server + agent + transport

## Errors encountered and resolutions
1) **Jupyter + stdio failed (`fileno` / async subprocess)**
   - Fix: used HTTP (`streamable-http`) transport instead of stdio.
2) **404 on server root**
   - Fix: MCP endpoint is `/mcp` (e.g., `http://127.0.0.1:8000/mcp`).
3) **Missing helper methods in adapter (`get_langchain_tools`, `read_resource`, `get_langchain_resources`)**
   - Fix: used `get_tools()` for tool conversion and MCP SDK (`ClientSession.read_resource`) for reading resources.

## Recommendation
- For scalable and reusable agent tooling, prefer MCP.
- For small one-off scripts, direct API integration is simpler.