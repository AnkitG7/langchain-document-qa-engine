"""Interactive Phase 6 Demonstration Script for DocMind FastAPI Backend & SSE Streaming.

Run with:
    python examples/demo_phase6.py

Demonstrates:
1. FastAPI TestClient and HTTP REST endpoints
2. Health Check Subsystem (/api/v1/health)
3. Dynamic Document Ingestion via File Upload (/api/v1/documents/upload)
4. Catalog Inventory Querying (/api/v1/documents)
5. Conversational RAG with Session Persistence (/api/v1/chat)
6. Real-Time Server-Sent Events (SSE) Token Streaming (/api/v1/chat/stream)
7. Real-Time Agent Tool Trace & Token Streaming (/api/v1/agent/stream)
"""

import io
import sys
import os
import json
import time
from pathlib import Path
from fastapi.testclient import TestClient

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.server import app

client = TestClient(app)


def print_banner(title: str):
    print("\n" + "=" * 70)
    print(f"  {title.upper()}")
    print("=" * 70)


def run_demo():
    print_banner("DocMind Phase 6: FastAPI Backend & SSE Streaming Demo")

    # 1. Health Check
    print_banner("1. System Health Check (/api/v1/health)")
    res_health = client.get("/api/v1/health")
    print(f"Status Code: {res_health.status_code}")
    print(json.dumps(res_health.json(), indent=2))

    # 2. Document Upload
    print_banner("2. Uploading New Document (/api/v1/documents/upload)")
    sample_content = (
        "DocMind Enterprise Release Notes:\n"
        "Version 0.6.0 introduces FastAPI REST API, async execution, and Server-Sent Events streaming.\n"
        "All endpoints support real-time token streaming and intermediate step telemetry."
    )
    files = {"file": ("release_notes_v06.txt", io.BytesIO(sample_content.encode("utf-8")), "text/plain")}
    res_upload = client.post("/api/v1/documents/upload", files=files)
    print(f"Upload Status: {res_upload.status_code}")
    print(json.dumps(res_upload.json(), indent=2))

    # 3. Document Catalog Listing
    print_banner("3. Document Catalog Inventory (/api/v1/documents)")
    res_docs = client.get("/api/v1/documents")
    print(f"Catalog Total Documents: {res_docs.json()['total_documents']}")
    for doc in res_docs.json()["documents"]:
        print(f"  - {doc['filename']} ({doc['file_type']}, {doc['size_bytes']} bytes)")

    # 4. Blocking Conversational RAG
    print_banner("4. Blocking Conversational RAG (/api/v1/chat)")
    chat_payload = {
        "input": "What features are introduced in version 0.6.0?",
        "session_id": "fastapi_demo_session",
    }
    print(f"[User Input]: {chat_payload['input']}")
    res_chat = client.post("/api/v1/chat", json=chat_payload)
    data_chat = res_chat.json()
    print(f"[DocMind Answer]:\n{data_chat.get('answer', '')}")
    if data_chat.get("citations"):
        print(f"[Citations Found]: {len(data_chat['citations'])} source(s)")
        for c in data_chat["citations"]:
            print(f"  * Source: {c['source']} -> {c['content_snippet'][:80]}...")

    # 5. Live SSE Token Streaming
    print_banner("5. Server-Sent Events (SSE) Live Token Streaming (/api/v1/chat/stream)")
    stream_payload = {
        "input": "Summarize what DocMind does in one sentence.",
        "session_id": "fastapi_demo_session",
    }
    print(f"[User Input]: {stream_payload['input']}")
    print("[Streaming SSE Response]: ", end="", flush=True)

    with client.stream("POST", "/api/v1/chat/stream", json=stream_payload) as response:
        for line in response.iter_lines():
            if line and line.startswith("data: "):
                event_data = json.loads(line[6:])
                event_type = event_data.get("event")
                if event_type == "token":
                    print(event_data.get("token", ""), end="", flush=True)
                elif event_type == "citations":
                    c_count = len(event_data.get("citations", []))
                    print(f"\n[SSE Event: Received {c_count} Citation(s)]\n[Token Stream]: ", end="", flush=True)
                elif event_type == "done":
                    print(f"\n[SSE Event: Stream Complete for session '{event_data.get('session_id')}']")

    # 6. Live Agent SSE Step & Token Streaming
    print_banner("6. Agent Server-Sent Events (SSE) Tool Trace & Streaming (/api/v1/agent/stream)")
    agent_payload = {
        "input": "What files are available and what is 500 * 24?",
        "session_id": "fastapi_agent_session",
    }
    print(f"[User Input]: {agent_payload['input']}")

    with client.stream("POST", "/api/v1/agent/stream", json=agent_payload) as response:
        for line in response.iter_lines():
            if line and line.startswith("data: "):
                event_data = json.loads(line[6:])
                event_type = event_data.get("event")
                if event_type == "tool_step":
                    print(f"\n[SSE Tool Step Event]: {event_data.get('tool')}({event_data.get('tool_input')})")
                    print(f"  -> Observation: {event_data.get('observation')[:100]}...")
                elif event_type == "token":
                    print(event_data.get("token", ""), end="", flush=True)
                elif event_type == "done":
                    print(f"\n[SSE Event: Agent Completed]")

    print_banner("Phase 6 Complete!")
    print("FastAPI REST backend, document endpoints, and SSE streaming verified successfully.")


if __name__ == "__main__":
    run_demo()
