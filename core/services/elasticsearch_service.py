# app/elasticsearch_service.py
from elasticsearch import Elasticsearch, helpers
from elasticsearch.exceptions import ConnectionError, RequestError
from datetime import datetime
import os
import time

class ElasticsearchService:
    def __init__(self, host="localhost", port=9200, index_name="documents"):
        self.client = Elasticsearch([f"http://{host}:{port}"], request_timeout=60)
        self.index_name = index_name

    def setup_index(self):
        """Tạo index với mapping linh hoạt"""
        try:
            self.client.indices.get(index=self.index_name)
            print(f"Index '{self.index_name}' đã tồn tại.")
        except Exception:
            # Mapping linh hoạt cho mọi loại content
            time.sleep(1.0)
            exists = self.client.indices.exists(index=self.index_name)

            if exists:
                print(f"Index '{self.index_name}' đã tồn tại.")
                self._wait_index_ready()
                return

            mapping = {
                "settings": {
                    "analysis": {
                        "analyzer": {
                            "multilingual_analyzer": {
                                "type": "custom",
                                "tokenizer": "standard",
                                "filter": ["lowercase", "asciifolding", "stop"]
                            },
                            "exact_analyzer": {
                                "type": "custom",
                                "tokenizer": "keyword",
                                "filter": ["lowercase"]
                            }
                        }
                    }
                },
                "mappings": {
                    "dynamic": True,  # Cho phép thêm field tự động
                    "properties": {
                        "content": {
                            "type": "text",
                            "analyzer": "multilingual_analyzer",
                            "fields": {
                                "exact": {
                                    "type": "text",
                                    "analyzer": "exact_analyzer"
                                }
                            }
                        },
                        "file_path": {"type": "keyword"},
                        "content_type": {"type": "keyword"},
                        "created_at": {"type": "date"}
                    }
                }
            }

            self.client.indices.create(index=self.index_name, body=mapping)
            print(f"Index '{self.index_name}' được tạo mới.")

    def _wait_index_ready(self, timeout=30):
        try:
            self.client.cluster.health(
                index=self.index_name,
                wait_for_status="yellow",
                request_timeout=timeout
            )
        except Exception as e:
            print(f"[ES] wait_index_ready warn: {e}")

    def _ensure_index(self):
        try:
            if not self.client.indices.exists(index=self.index_name):
                self.setup_index()
        except Exception:
            # phòng khi ES chưa trả lời kịp
            time.sleep(1.0)
            if not self.client.indices.exists(index=self.index_name):
                self.setup_index()

    def upsert_documents(self, documents):
        """Bulk upsert documents"""
        if not documents:
            return

        self._ensure_index()

        actions = []
        for doc in documents:
            action = {
                "_index": self.index_name,
                "_id": doc.get("id", self._generate_id(doc.get("file_path", ""))),
                "_source": {
                    **doc,
                    "created_at": datetime.now().isoformat()
                }
            }
            actions.append(action)

        try:
            helpers.bulk(self.client, actions, refresh=True)
            print(f"Upserted {len(actions)} documents")
        except Exception as e:
            print(f"Lỗi upsert: {e}")

    def _has_ngram_field(self) -> bool:
        """
        Kiểm tra (và cache) xem index có field 'content.ngram' không.
        Nếu không có hoặc lỗi API thì trả False.
        """
        if getattr(self, "_has_ngram", None) is not None:
            return self._has_ngram
        try:
            fm = self.client.indices.get_field_mapping(
                index=self.index_name,
                fields="content.ngram"
            )
            # fm dạng { "<index>": {"mappings": {"content.ngram": {...}}}}
            self._has_ngram = any(
                "content.ngram" in (v.get("mappings") or {})
                for v in (fm or {}).values()
            )
        except Exception:
            self._has_ngram = False
        return self._has_ngram

    def search(self, query, limit=10, score_threshold=0.1, filters=None):
        """
        Search giữ nguyên độ chính xác:
        - match_phrase (boost cao)
        - match AND
        - match OR (min_should_match)
        - fuzzy AUTO (nhẹ tay)
        - "chứa cụm từ": ưu tiên content.ngram nếu có; nếu không, fallback wildcard như cũ
        Tối ưu hiệu năng vô hại:
        - track_total_hits=False (không ảnh hưởng thứ hạng/kết quả top-N)
        """
        import time

        start = time.time()
        q = (query or "").strip()
        if not q:
            return []

        # Các chiến lược đang dùng
        should_clauses = [
            # Exact phrase
            {"match_phrase": {"content": {"query": q, "boost": 8.0}}},

            # Tất cả từ
            {"match": {"content": {"query": q, "operator": "and", "boost": 5.0}}},

            # Một phần từ, bắt buộc 60% từ khớp
            {"match": {
                "content": {
                    "query": q,
                    "operator": "or",
                    "minimum_should_match": "60%",
                    "boost": 3.0
                }
            }},

            # Fuzzy cho lỗi chính tả (nhẹ tay)
            {"match": {
                "content": {
                    "query": q,
                    "fuzziness": "AUTO",
                    "prefix_length": 1,
                    "boost": 2.0
                }
            }}
        ]

        # "Chứa cụm từ": nếu có n-gram thì dùng n-gram (nhanh hơn),
        # nếu chưa có thì fallback wildcard để KHÔNG thay đổi hành vi hiện tại.
        if self._has_ngram_field():
            should_clauses.append({
                "match": {
                    "content.ngram": {
                        "query": q,
                        "operator": "and",
                        "boost": 2.5
                    }
                }
            })
        else:
            should_clauses.append({
                "wildcard": {
                    "content": {
                        "value": f"*{q}*",
                        "boost": 1.0
                    }
                }
            })

        # Ghép bộ lọc (nếu có)
        bool_query = {
            "should": should_clauses,
            "minimum_should_match": 1
        }
        if filters:
            filter_clauses = [{"term": {k: v}} for k, v in filters.items()]
            if filter_clauses:
                bool_query["filter"] = filter_clauses

        search_body = {
            "track_total_hits": False,              # giảm chi phí, không ảnh hưởng top-N
            "query": {"bool": bool_query},
            "size": limit,
            "min_score": score_threshold,
            "highlight": {
                "fields": {
                    "content": {
                        "pre_tags": ["<mark>"],
                        "post_tags": ["</mark>"],
                        "fragment_size": 150,
                        "number_of_fragments": 3
                    }
                }
            }
        }

        try:
            response = self.client.search(index=self.index_name, body=search_body)
            results = []
            for hit in response.get("hits", {}).get("hits", []):
                item = {
                    "id": hit.get("_id"),
                    "score": hit.get("_score"),
                    **(hit.get("_source") or {})
                }
                # giữ highlight như cũ
                hl = []
                for _, frags in (hit.get("highlight") or {}).items():
                    hl.extend(frags)
                if hl:
                    item["highlights"] = hl
                results.append(item)
            print(f"[ES] search '{q}' returned {len(results)} hits in {time.time() - start:.3f}s")
            return results

        except Exception as e:
            print(f"Lỗi search: {e}")
            return []

    def search_exact_phrase(self, query, limit=10, filters=None, score_threshold=None):
        """
        Tìm exact-match kiểu 'chứa cụm từ' (substring) không phân biệt hoa/thường.
        Yêu cầu: content.exact đã mapping với analyzer=exact_analyzer (tokenizer=keyword + lowercase).
        """

        # Chuẩn hoá truy vấn theo đúng analyzer (lowercase)
        q = (query or "").strip().lower()
        if not q:
            return []

        # Wildcard *q* để kiểm tra "chứa cụm từ"
        must_clause = [{"wildcard": {"content.exact": {"value": f"*{q}*"}}}]

        # Thêm filters (term) nếu có
        filter_clauses = []
        if filters:
            for key, value in filters.items():
                filter_clauses.append({"term": {key: value}})

        search_body = {
            "query": {
                "bool": {
                    "must": must_clause,
                    **({"filter": filter_clauses} if filter_clauses else {}),
                }
            },
            "size": limit,
            # min_score không hữu ích lắm với wildcard, nhưng giữ tham số cho thống nhất API
            **({"min_score": score_threshold} if score_threshold is not None else {}),
            "highlight": {
                "fields": {
                    # highlight hiển thị trên content (đã tách từ) để người dùng dễ đọc
                    "content": {
                        "pre_tags": ["<mark>"],
                        "post_tags": ["</mark>"],
                        "fragment_size": 150,
                        "number_of_fragments": 3,
                    }
                }
            },
        }

        try:
            resp = self.client.search(index=self.index_name, body=search_body)

            results = []
            for hit in resp.get("hits", {}).get("hits", []):
                item = {
                    "id": hit.get("_id"),
                    "score": hit.get("_score"),
                    **hit.get("_source", {}),
                }

                if "highlight" in hit:
                    hl = []
                    for _, frags in hit["highlight"].items():
                        hl.extend(frags)
                    item["highlights"] = hl

                results.append(item)

            return results

        except Exception as e:
            print(f"Lỗi search_exact_phrase: {e}")
            return []

    def _generate_id(self, file_path):
        """Sinh _id ổn định theo MD5(file_path); không bao giờ rỗng."""
        import hashlib, os
        fp = (file_path or "").strip()
        if not fp:
            return hashlib.md5(os.urandom(16)).hexdigest()
        return hashlib.md5(fp.encode("utf-8", errors="ignore")).hexdigest()

    def get_stats(self):
        """Lấy stats của index"""
        try:
            if not self.client.indices.exists(index=self.index_name):
                return {"document_count": 0, "index_size_mb": 0}

            stats = self.client.indices.stats(index=self.index_name)
            doc_count = stats['indices'][self.index_name]['total']['docs']['count']
            size = stats['indices'][self.index_name]['total']['store']['size_in_bytes']

            return {
                "document_count": doc_count,
                "index_size_mb": round(size / (1024 * 1024), 2)
            }
        except:
            return {"document_count": 0, "index_size_mb": 0}

# Singleton instance
es_service = ElasticsearchService()
