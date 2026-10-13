import sys
import json
import urllib.request

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_req = urllib.request.Request(
    "http://127.0.0.1:8000/api/load-sample",
    data=json.dumps({"sample_key": "transformer_35_phut_vi"}).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)
with urllib.request.urlopen(load_req) as resp:
    res = json.loads(resp.read().decode("utf-8"))
    print(f"Loaded lecture: {res['title']} ({res['formatted_duration']}), has_video: {res['has_video']}")

query = "Token và embedding vector trong Transformer là gì?"
print(f"\nQuery: '{query}'")

rag_req = urllib.request.Request(
    "http://127.0.0.1:8000/api/rag/ask",
    data=json.dumps({
        "query": query,
        "alpha": 0.7,
        "use_query_expansion": True
    }).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)
with urllib.request.urlopen(rag_req) as resp:
    rag_res = json.loads(resp.read().decode("utf-8"))
    pg = rag_res.get("primary_grounding", {})
    print("\n=== KẾT QUẢ TÌM KIẾM VIDEO GROUNDING ===")
    print(f"Phân đoạn: {pg.get('start_str')} -> {pg.get('end_str')}")
    print(f"Độ tin cậy: {pg.get('confidence')}")
    print(f"Tiêu đề: {pg.get('slide_title')}")
    print("\nTrích dẫn minh chứng:")
    for c in rag_res.get("evidence_citations", [])[:4]:
        print(f"  [{c['timestamp_str']}]: \"{c['quote']}\"")
