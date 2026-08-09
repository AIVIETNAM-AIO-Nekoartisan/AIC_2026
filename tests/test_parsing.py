# Unit tests cho utils/parsing.py — chạy: python3 -m unittest discover -s tests
import unittest

from core.utils.parsing import parse_hms_to_seconds, extract_frame_id_from_payload


class TestParseHmsToSeconds(unittest.TestCase):
    def test_hh_mm_ss(self):
        self.assertEqual(parse_hms_to_seconds("01:02:03"), 3723.0)

    def test_hh_mm_ss_millis(self):
        self.assertAlmostEqual(parse_hms_to_seconds("01:02:03.5"), 3723.5)

    def test_mm_ss(self):
        self.assertEqual(parse_hms_to_seconds("02:03"), 123.0)

    def test_whitespace(self):
        self.assertEqual(parse_hms_to_seconds("  00:01:00 "), 60.0)

    def test_invalid(self):
        self.assertIsNone(parse_hms_to_seconds("abc"))
        self.assertIsNone(parse_hms_to_seconds("1:2:3:4"))
        self.assertIsNone(parse_hms_to_seconds(""))
        self.assertIsNone(parse_hms_to_seconds(None))


class TestExtractFrameId(unittest.TestCase):
    def test_from_frame_index(self):
        self.assertEqual(extract_frame_id_from_payload({"frame": {"index": 42}}), 42)

    def test_frame_index_string(self):
        self.assertEqual(extract_frame_id_from_payload({"frame": {"index": "7"}}), 7)

    def test_index_wins_over_filename(self):
        payload = {"frame": {"index": 1}, "filename": "x__frame_000999.webp"}
        self.assertEqual(extract_frame_id_from_payload(payload), 1)

    def test_fallback_filename(self):
        payload = {"filename": "L21_V003__shot_009__pos_middle__frame_000600__ts_20.0.webp"}
        self.assertEqual(extract_frame_id_from_payload(payload), 600)

    def test_fallback_file_path(self):
        payload = {"file_path": "/data/L01_V001__frame_000123.webp"}
        self.assertEqual(extract_frame_id_from_payload(payload), 123)

    def test_bad_index_falls_back_to_filename(self):
        payload = {"frame": {"index": "n/a"}, "filename": "a__frame_000005.jpg"}
        self.assertEqual(extract_frame_id_from_payload(payload), 5)

    def test_missing(self):
        self.assertIsNone(extract_frame_id_from_payload({}))
        self.assertIsNone(extract_frame_id_from_payload(None))
        self.assertIsNone(extract_frame_id_from_payload({"filename": "no_frame_here.jpg"}))


if __name__ == "__main__":
    unittest.main()
