# Integration tests — cần stack đang chạy; tự skip nếu service không lên.
# Chạy: python3 -m unittest discover -s tests -v
import json
import unittest
import urllib.error
import urllib.request

API_URL = "http://localhost:8000"
QDRANT_URL = "http://localhost:6333"
ES_URL = "http://localhost:9200"


def _get(url: str, timeout: float = 5.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode())


def _up(url: str) -> bool:
    try:
        urllib.request.urlopen(url, timeout=2)
        return True
    except Exception:
        return False


@unittest.skipUnless(_up(f"{API_URL}/health"), "API server không chạy tại :8000")
class TestApiServer(unittest.TestCase):
    def test_health(self):
        status, body = _get(f"{API_URL}/health")
        self.assertEqual(status, 200)

    def test_kis_search_returns_results_schema(self):
        req = urllib.request.Request(
            f"{API_URL}/search/kis",
            data=json.dumps({"text_query": "a person walking", "num_results": 3}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            body = json.loads(r.read().decode())
        self.assertTrue(body["success"])
        self.assertIsInstance(body["results"], list)
        self.assertIn("total_count", body)

    def test_ocr_search_schema(self):
        req = urllib.request.Request(
            f"{API_URL}/search/ocr",
            data=json.dumps({"query": "test", "num_results": 3}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            body = json.loads(r.read().decode())
        self.assertIn("results", body)


@unittest.skipUnless(_up(f"{QDRANT_URL}/collections"), "Qdrant không chạy tại :6333")
class TestQdrant(unittest.TestCase):
    def test_collections_endpoint(self):
        status, body = _get(f"{QDRANT_URL}/collections")
        self.assertEqual(status, 200)
        self.assertIn("result", body)


@unittest.skipUnless(_up(ES_URL), "Elasticsearch không chạy tại :9200")
class TestElasticsearch(unittest.TestCase):
    def test_cluster_reachable(self):
        status, body = _get(ES_URL)
        self.assertEqual(status, 200)
        self.assertIn("version", body)


if __name__ == "__main__":
    unittest.main()
