import unittest
from backend.vision.vision_extractor import vision_extractor
from backend.config import KEYFRAMES_DIR

class TestVisionExtractor(unittest.TestCase):
    def test_singleton_initialization(self):
        self.assertIsNotNone(vision_extractor)

    def test_nonexistent_video_returns_empty(self):
        kfs = vision_extractor.extract_slide_keyframes(
            video_path="nonexistent_video.mp4",
            video_id="mock_video"
        )
        self.assertEqual(kfs, [])

    def test_get_keyframe_for_timestamp(self):
        mock_keyframes = [
            {"timestamp": 10.0, "timestamp_str": "00:10", "slide_title": "Intro"},
            {"timestamp": 60.0, "timestamp_str": "01:00", "slide_title": "Architecture"},
            {"timestamp": 120.0, "timestamp_str": "02:00", "slide_title": "Conclusion"}
        ]
        kf = vision_extractor.get_keyframe_for_timestamp(mock_keyframes, 75.0)
        self.assertIsNotNone(kf)
        self.assertEqual(kf["slide_title"], "Architecture")

        kf_start = vision_extractor.get_keyframe_for_timestamp(mock_keyframes, 5.0)
        self.assertEqual(kf_start["slide_title"], "Intro")

if __name__ == "__main__":
    unittest.main()
