#Agentic AI
"""
main.py
───────
Entry point for the AI Research Agent.
Run this file to verify your environment is set up correctly.

Usage:
    python main.py
"""

import sys
from config import validate_env

def main():
    print("=" * 60)
    print("  🤖  AI Research & Business Reporting Agent")
    print("=" * 60)

    # Step 1: Validate all API keys are present
    print("\n[1/3] Checking environment variables...")
    try:
        validate_env()
        print("      ✅ All API keys found.")
    except EnvironmentError as e:
        print(f"      ⚠️  {e}")
        sys.exit(1)

    # Step 2: Confirm project structure
    print("\n[2/3] Verifying project structure...")
    import os
    expected_dirs = ["nodes", "utils", "outputs", "tests"]
    for d in expected_dirs:
        status = "✅" if os.path.isdir(d) else "❌ MISSING"
        print(f"      {status}  ./{d}/")

    # Step 3: Ready message
    print("\n[3/3] Environment ready.")
    print("\n  Next step → Step 2: Data Ingestion (Tavily + ArXiv)")
    print("=" * 60)


if __name__ == "__main__":
    main()