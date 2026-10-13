import json
import csv
import io
import time
from pathlib import Path
from typing import List, Dict, Optional
from backend.config import ANNOTATIONS_DIR

ANNOTATIONS_FILE = ANNOTATIONS_DIR / "user_annotations.json"

class AnnotationManager:
    """
    Manages user ground truth annotations and evaluation feedback.
    """
    def __init__(self):
        self.file_path = ANNOTATIONS_FILE
        self._ensure_file()

    def _ensure_file(self):
        if not self.file_path.exists():
            self.file_path.write_text("[]", encoding="utf-8")

    def load_annotations(self) -> List[Dict]:
        try:
            content = self.file_path.read_text(encoding="utf-8")
            return json.loads(content)
        except Exception:
            return []

    def add_annotation(
        self,
        video_id: str,
        query: str,
        start: float,
        end: float,
        rating: str, # "correct", "partial", "incorrect"
        notes: str = "",
        predicted_start: Optional[float] = None,
        predicted_end: Optional[float] = None
    ) -> Dict:
        annotations = self.load_annotations()
        item = {
            "id": f"ann_{int(time.time() * 1000)}",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "video_id": video_id,
            "query": query.strip(),
            "start": round(float(start), 2),
            "end": round(float(end), 2),
            "duration": round(float(end) - float(start), 2),
            "rating": rating,
            "notes": notes.strip(),
            "predicted_start": round(float(predicted_start), 2) if predicted_start is not None else None,
            "predicted_end": round(float(predicted_end), 2) if predicted_end is not None else None
        }
        annotations.append(item)
        self.file_path.write_text(json.dumps(annotations, ensure_ascii=False, indent=2), encoding="utf-8")
        return item

    def delete_annotation(self, annotation_id: str) -> bool:
        annotations = self.load_annotations()
        filtered = [a for a in annotations if a["id"] != annotation_id]
        if len(filtered) != len(annotations):
            self.file_path.write_text(json.dumps(filtered, ensure_ascii=False, indent=2), encoding="utf-8")
            return True
        return False

    def export_csv(self) -> str:
        annotations = self.load_annotations()
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["ID", "Timestamp", "Video ID", "Query", "Start (s)", "End (s)", "Duration (s)", "Rating", "Pred Start", "Pred End", "Notes"])
        for a in annotations:
            writer.writerow([
                a.get("id"),
                a.get("timestamp"),
                a.get("video_id"),
                a.get("query"),
                a.get("start"),
                a.get("end"),
                a.get("duration"),
                a.get("rating"),
                a.get("predicted_start"),
                a.get("predicted_end"),
                a.get("notes")
            ])
        return output.getvalue()

annotation_manager = AnnotationManager()
