import unittest
from backend.audio.subtitle_parser import timestamp_to_seconds, seconds_to_timestamp, parse_srt_or_vtt
from backend.config import SAMPLES_DIR

class TestSubtitleParser(unittest.TestCase):
    def test_timestamp_conversions(self):
        self.assertEqual(timestamp_to_seconds("00:01:30,500"), 90.5)
        self.assertEqual(timestamp_to_seconds("01:00:00.000"), 3600.0)
        self.assertEqual(timestamp_to_seconds("02:15"), 135.0)
        self.assertEqual(seconds_to_timestamp(90.5), "01:30")
        self.assertEqual(seconds_to_timestamp(3665.0), "01:01:05")

    def test_parse_sample_srt(self):
        sample_file = SAMPLES_DIR / "machine_learning_intro_vi.srt"
        self.assertTrue(sample_file.exists())
        subs = parse_srt_or_vtt(sample_file)
        self.assertGreaterEqual(len(subs), 20)
        self.assertIn("Overfitting", subs[0]["text"])

if __name__ == "__main__":
    unittest.main()
