from pathlib import Path
from typing import List
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("FilesystemServer")

# Use working directory instead of __file__ (safer)
ROOT = Path.cwd() / "sandbox_files"
ROOT.mkdir(exist_ok=True)


def _safe_path(rel_path: str) -> Path:
    p = (ROOT / rel_path).resolve()
    if ROOT.resolve() not in p.parents and p != ROOT.resolve():
        raise ValueError("Path escapes sandbox root.")
    return p


@mcp.tool()
def list_dir(rel_dir: str = ".") -> List[str]:
    d = _safe_path(rel_dir)
    return sorted([p.name for p in d.iterdir()])


@mcp.tool()
def read_text(rel_path: str) -> str:
    return _safe_path(rel_path).read_text(encoding="utf-8")


@mcp.tool()
def write_text(rel_path: str, content: str) -> str:
    p = _safe_path(rel_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return "File written successfully"

if __name__ == "__main__":
    mcp.run(transport="http")
