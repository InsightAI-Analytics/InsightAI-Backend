"""Quick integration smoke test for the InsightAI backend."""
import urllib.request
import json
import uuid

BASE = "http://localhost:8000"
SESSION = str(uuid.uuid4())
print(f"Testing with session: {SESSION}")

# --- Health check ---
with urllib.request.urlopen(f"{BASE}/health") as r:
    health = json.loads(r.read())
    assert health["status"] == "ok", f"Health failed: {health}"
    print("✓ Health check passed")

# --- Upload file ---
boundary = "boundary12345insightai"
with open("samples/sales_data.csv", "rb") as f:
    file_content = f.read()

body = (
    b"--" + boundary.encode() + b"\r\n"
    b'Content-Disposition: form-data; name="files"; filename="sales_data.csv"\r\n'
    b"Content-Type: text/csv\r\n\r\n"
    + file_content
    + b"\r\n--" + boundary.encode() + b"--\r\n"
)

req = urllib.request.Request(
    f"{BASE}/api/files/upload",
    data=body,
    headers={
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "X-Session-ID": SESSION,
    },
    method="POST",
)
with urllib.request.urlopen(req) as r:
    upload_data = json.loads(r.read())

assert len(upload_data["uploaded"]) == 1, f"Expected 1 uploaded file, got: {upload_data}"
assert upload_data["uploaded"][0]["rows"] == 24, f"Expected 24 rows: {upload_data['uploaded'][0]}"
print(f"✓ Upload passed: {upload_data['uploaded'][0]['name']} ({upload_data['uploaded'][0]['rows']} rows, {upload_data['uploaded'][0]['columns']} columns)")
print(f"  Errors: {upload_data['errors']}")

# --- List files ---
req2 = urllib.request.Request(
    f"{BASE}/api/files",
    headers={"X-Session-ID": SESSION},
)
with urllib.request.urlopen(req2) as r:
    files_data = json.loads(r.read())

assert len(files_data) == 1
print(f"✓ List files passed: {len(files_data)} file(s)")

# --- Query (no LLM key expected in test env, check graceful error) ---
query_body = json.dumps({"session_id": SESSION, "question": "What is the total revenue?"}).encode()
req3 = urllib.request.Request(
    f"{BASE}/api/query",
    data=query_body,
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req3) as r:
    query_data = json.loads(r.read())

assert "answer" in query_data, f"No answer in response: {query_data}"
print(f"✓ Query endpoint responded (answer length: {len(query_data['answer'])} chars)")
if query_data.get("error"):
    print(f"  Note: LLM returned error (expected without API key): {query_data['error'][:100]}")
elif query_data.get("metrics"):
    print(f"  Metrics: {query_data['metrics']}")

print("\n✅ All smoke tests passed!")
