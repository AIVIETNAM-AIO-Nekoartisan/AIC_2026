# app/vector_store_optimized.py
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, NamedVector, Range as QRange
import re
from typing import List, Dict, Optional, Union
import asyncio
import numpy as np
import time
import logging
from functools import lru_cache
import os
from concurrent.futures import ThreadPoolExecutor
import uuid

# Setup logging
logger = logging.getLogger(__name__)

from core.utils.parsing import extract_frame_id_from_payload as _extract_frame_id_from_payload


class QdrantService:
    """
    Optimized Qdrant service with auto local/cloud detection
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        collection_name: str = "test-aic",
        vector_dim: int = 1024,
        qdrant_url: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        named_vectors: Optional[Dict[str, int]] = None
    ):

        self.api_key = api_key
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        self.timeout = timeout
        self.max_retries = max_retries
        self.named_vectors = named_vectors

        # 🔧 AUTO CONNECTION: Local if no API key, cloud if has API key
        self.client = None
        self._connect_to_qdrant(qdrant_url)

        # Thread pool for concurrent operations
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Result caching
        self._search_cache = {}
        self.cache_ttl = 300  # 5 minutes cache
        self.max_cache_size = 1000

        print(f"✅ QdrantService initialized: {collection_name} (dim={vector_dim})")

    def _connect_to_qdrant(self, qdrant_url: Optional[str]):
        """Connect to Qdrant - cloud if API key, local if not"""

        # If has API key, try cloud first
        if self.api_key:
            try:
                cloud_url = qdrant_url or "https://d9a11d63-c516-4805-b00a-e6a4bbf1f282.us-west-2-0.aws.cloud.qdrant.io"
                self.client = QdrantClient(url=cloud_url, api_key=self.api_key)
                collections = self.client.get_collections()
                print(f"✅ Connected to Qdrant Cloud - {len(collections.collections)} collections found")
                return
            except Exception as cloud_error:
                print(f"⚠️ Cloud connection failed: {cloud_error}")

        # Try local connection
        local_url = qdrant_url or "http://localhost:6333"
        try:
            self.client = QdrantClient(url=local_url)
            collections = self.client.get_collections()
            print(f"✅ Connected to Local Qdrant - {len(collections.collections)} collections found")
            return
        except Exception as local_error:
            print(f"⚠️ Local connection failed: {local_error}")

        # Fallback to in-memory
        try:
            self.client = QdrantClient(":memory:")
            print("✅ Using in-memory Qdrant (data will not persist)")
        except Exception as memory_error:
            print(f"❌ In-memory Qdrant failed: {memory_error}")
            print("🔄 Using mock Qdrant client for development...")
            # self.client = MockQdrantClient()

    def _get_cache_key(self, vector: List[float], limit: int, score_threshold: float, filter_condition=None) -> str:
        """Generate cache key for search results"""
        import hashlib
        vector_hash = hashlib.md5(str(vector[:10]).encode()).hexdigest()[:8]
        flt = hashlib.md5(str(filter_condition).encode()).hexdigest()[:8] if filter_condition else ""
        return f"{vector_hash}_{limit}_{score_threshold}_{flt}"

    def _cleanup_cache(self):
        """Clean up expired cache entries"""
        current_time = time.time()
        expired_keys = [
            key
            for key, (result, timestamp) in self._search_cache.items()
            if current_time - timestamp > self.cache_ttl
        ]
        for key in expired_keys:
            del self._search_cache[key]

        # Limit cache size
        if len(self._search_cache) > self.max_cache_size:
            items_to_remove = len(self._search_cache) // 5
            keys_to_remove = list(self._search_cache.keys())[:items_to_remove]
            for key in keys_to_remove:
                del self._search_cache[key]

    def setup_collection(self):
        try:
            try:
                collection_info = self.client.get_collection(
                    collection_name=self.collection_name
                )
                print(f"✅ Collection '{self.collection_name}' exists with {collection_info.points_count} points")
            except Exception:
                print(f"🔧 Creating collection '{self.collection_name}'...")
                if self.named_vectors:
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config={
                            name: VectorParams(size=dim, distance=Distance.COSINE)
                            for name, dim in self.named_vectors.items()
                        },
                    )
                else:
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE),
                    )
                print(f"✅ Collection '{self.collection_name}' created successfully")

            # 🔑 BẮT BUỘC CHO order_by:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="frame.timestamp_seconds",
                    field_schema=models.PayloadSchemaType.FLOAT,   # hoặc "float"
                )
                print("✅ Index OK: frame.timestamp_seconds (float)")
            except Exception as e:
                print(f"ℹ️  frame.timestamp_seconds index: {e}")

            # (nên) index luôn video.name để filter nhanh
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="video.name",
                    field_schema=models.PayloadSchemaType.KEYWORD, # hoặc "keyword"
                )
                print("✅ Index OK: video.name (keyword)")
            except Exception as e:
                print(f"ℹ️  video.name index: {e}")

            return True
        except Exception as e:
            print(f"❌ Collection setup failed: {e}")
            return False

    def search(
        self,
        vector: Union[List[float], np.ndarray],
        limit: int = 10,
        score_threshold: float = 0.0,
        use_cache: bool = True,
        filter_condition: Optional[Filter] = None,
    ) -> List:
        """
        Search with error handling and fallbacks
        """
        start_time = time.time()

        # Convert vector to list if numpy array
        if isinstance(vector, np.ndarray):
            vector = vector.tolist()

        # Check cache
        cache_key = self._get_cache_key(vector, limit, score_threshold, filter_condition) if use_cache else None

        if cache_key and cache_key in self._search_cache:
            result, timestamp = self._search_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return result

        # Perform search
        try:
            # qdrant-client >= 1.13: search() bị bỏ, dùng query_points()
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=vector,
                limit=limit,
                with_payload=True,
                with_vectors=False,
                score_threshold=score_threshold,
                query_filter=filter_condition
            ).points

            elapsed = time.time() - start_time
            logger.info(f"Search completed: {len(search_result)} results in {elapsed:.3f}s")

            # Cache the result
            if cache_key:
                self._search_cache[cache_key] = (search_result, time.time())
                self._cleanup_cache()

            return search_result

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def search_by(
    self,
    vector_name: str,
    vector: Union[List[float], np.ndarray],
    limit: int = 10,
    score_threshold: float = 0.0,
    filter_condition: Optional[Filter] = None,
) -> List:
        if isinstance(vector, np.ndarray):
            vector = vector.tolist()
        try:
            return self.client.query_points(
                collection_name=self.collection_name,
                query=vector,
                using=vector_name,
                limit=limit,
                with_payload=True,
                with_vectors=False,
                score_threshold=score_threshold,
                query_filter=filter_condition
            ).points
        except Exception as e:
            logger.error(f"Named-vector search failed ({vector_name}): {e}")
            return []

    async def search_async(
        self,
        vector: Union[List[float], np.ndarray],
        limit: int = 10,
        score_threshold: float = 0.0,
        use_cache: bool = True,
    ) -> List:
        """Async version of search"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, self.search, vector, limit, score_threshold, use_cache
        )

    def upsert_points(self, points: List[PointStruct], batch_size: int = 100):
        """Upsert with error handling"""
        if not points:
            return

        try:
            total_points = len(points)
            print(f"🔄 Upserting {total_points} points...")

            for i in range(0, total_points, batch_size):
                batch = points[i : i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch,
                    wait=True
                )

            print(f"✅ Successfully upserted {total_points} points")

        except Exception as e:
            logger.error(f"Upsert failed: {e}")

    def delete_points(self, point_ids: List[str]):
        """Delete points by IDs"""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=point_ids,
                wait=True
            )
            return True
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return False

    def delete_by_filter(self, field: str, value: str):
        """Delete points by filter"""
        try:
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key=field,
                        match=MatchValue(value=value)
                    )
                ]
            )

            self.client.delete(
                collection_name=self.collection_name,
                points_selector=filter_condition,
                wait=True
            )
            return True
        except Exception as e:
            logger.error(f"Delete by filter failed: {e}")
            return False

    def get_collection_info(self) -> Dict:
        """Get collection info with error handling"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "points_count": info.points_count,
                "vector_dim": self.vector_dim,
                "status": info.status,
                "cache_size": len(self._search_cache),
                "connection_type": self._get_connection_type()
            }
        except Exception as e:
            return {
                "name": self.collection_name,
                "error": str(e),
                "cache_size": len(self._search_cache),
                # "mock_mode": isinstance(self.client, MockQdrantClient),
                "connection_type": self._get_connection_type()
            }

    def _get_connection_type(self) -> str:
        """Get connection type for debugging"""
        # if isinstance(self.client, MockQdrantClient):
        #     return "mock"
        if hasattr(self.client, '_client') and ':memory:' in str(self.client._client):
            return "memory"
        elif self.api_key:
            return "cloud"
        else:
            return "local"

    def health_check(self) -> Dict:
        """Health check with error handling"""
        try:
            start_time = time.time()
            collections = self.client.get_collections()
            connection_time = time.time() - start_time

            return {
                "status": "healthy",
                "connection_time": connection_time,
                "collections_count": len(collections.collections),
                "connection_type": self._get_connection_type(),
                "has_api_key": bool(self.api_key)
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "connection_type": self._get_connection_type(),
                "has_api_key": bool(self.api_key)
            }

    def get_video_shots_range(self, video_name: str, start_shot: int, end_shot: int) -> List[Dict]:
        """Lấy shots trong khoảng từ start_shot đến end_shot"""
        try:
            filter_condition = Filter(
                must=[
                    FieldCondition(key="video.name", match=MatchValue(value=video_name)),
                    FieldCondition(key="shot.number", range=QRange(gte=start_shot, lte=end_shot))
                ]
            )

            shots_data = {}
            next_page = None

            while True:
                points, next_page = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=filter_condition,
                    with_payload=True,
                    with_vectors=False,
                    limit=200,
                    offset=next_page
                )

                if not points:
                    break

                for point in points:
                    payload = point.payload or {}
                    shot = payload.get("shot") or {}
                    shot_number = shot.get("number")
                    position = shot.get("position")

                    if shot_number is not None and position:
                        shot_id = f"{video_name}_shot_{shot_number}"

                        if shot_id not in shots_data:
                            shots_data[shot_id] = {
                                "shot_id": shot_id,
                                "video_name": video_name,
                                "shot_number": shot_number,
                                "positions": {}
                            }

                        shots_data[shot_id]["positions"][position] = {
                            "file_path": payload.get("file_path"),
                            "frame_id": _extract_frame_id_from_payload(payload),
                            "timestamp": (payload.get("frame") or {}).get("timestamp_formatted"),
                            "timestamp_seconds": (payload.get("frame") or {}).get("timestamp_seconds"),
                            "payload": payload
                        }

                if not next_page:
                    break

            result = sorted(shots_data.values(), key=lambda x: x["shot_number"])
            return result

        except Exception as e:
            logger.error(f"get_video_shots_range failed: {e}")
            return []

    def get_frames_by_timerange(
    self,
    video_name: str,
    start_s: float,
    end_s: float,
    limit: int = 300,
    sample_stride: int = 1,
):
        """
        Trả về tối đa `limit` frames trong [start_s, end_s], đã sort theo timestamp tăng dần.
        - DỪNG SỚM khi đủ `limit` (không scroll hết collection).
        - Không lấy vectors.
        - Sử dụng order_by payload để server làm việc nặng.
        - Sampling on-the-fly bằng stride để giảm mật độ.
        """
        start_s = float(max(0.0, start_s))
        end_s = float(max(start_s, end_s))
        cap = max(1, int(limit))

        f = Filter(
            must=[
                FieldCondition(key="video.name", match=MatchValue(value=video_name)),
                FieldCondition(key="frame.timestamp_seconds", range=QRange(gte=start_s, lte=end_s)),
            ]
        )

        # Page nhỏ giúp latency thấp và có cơ hội dừng sớm
        page = min(200, cap)  # 200 là sweet spot
        acc, next_page = [], None
        stride = max(1, int(sample_stride))
        picked = 0

        while True:
            points, next_page = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=f,
                with_vectors=False,
                with_payload=True,
                limit=page,
                offset=next_page,
                order_by="frame.timestamp_seconds",   # Qdrant >= 1.6
            )

            if not points:
                break

            for idx, p in enumerate(points):
                # sampling stride: lấy mỗi stride-th item
                if ((len(acc) + idx) % stride) != 0:
                    continue

                payload = p.payload or {}
                frame = payload.get("frame") or {}
                ts = frame.get("timestamp_seconds")
                if ts is None:
                    # fallback nếu payload khác schema
                    ts = payload.get("timestamp_seconds")
                    if ts is None:
                        continue

                fp = payload.get("file_path")
                if not fp:
                    continue

                # frame_id (tùy schema)
                fid = frame.get("index")
                if fid is None:
                    # parse từ file_path nếu cần
                    import re, os
                    m = re.search(r"__frame_(\d+)", os.path.basename(fp))
                    fid = int(m.group(1)) if m else None

                acc.append({
                    "file_path": fp,
                    "payload": payload,
                    "frame_id": fid,
                    "timestamp_seconds": float(ts),
                    "timestamp_formatted": frame.get("timestamp_formatted"),
                })
                picked += 1
                if picked >= cap:
                    break

            if picked >= cap or next_page is None:
                break

        # sort lại đề phòng backend không giữ thứ tự tuyệt đối
        acc.sort(key=lambda x: x["timestamp_seconds"])
        return acc

    def update_shot_time_fields(self, video_name: Optional[str] = None, batch_size: int = 100):
        """
        Update all points to add start_time and end_time fields based on shot positions.

        Args:
            video_name: Optional video name to filter by. If None, processes all videos.
            batch_size: Number of points to process in each batch
        """
        try:
            print(f"🔄 Starting shot time fields update for video: {video_name or 'ALL'}")

            # Step 1: Get all shot data grouped by video and shot
            filter_condition = None
            if video_name:
                filter_condition = Filter(
                    must=[FieldCondition(key="video.name", match=MatchValue(value=video_name))]
                )

            # Collect all shots data
            shots_timing_data = {}  # {video_name: {shot_number: {start_time, end_time}}}
            next_page = None
            total_processed = 0

            print("📊 Phase 1: Collecting shot timing data...")
            while True:
                points, next_page = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=filter_condition,
                    with_payload=True,
                    with_vectors=False,
                    limit=batch_size,
                    offset=next_page
                )

                if not points:
                    break

                for point in points:
                    payload = point.payload or {}
                    video = payload.get("video") or {}
                    shot = payload.get("shot") or {}
                    frame = payload.get("frame") or {}

                    v_name = video.get("name")
                    shot_number = shot.get("number")
                    position = shot.get("position")
                    timestamp_seconds = frame.get("timestamp_seconds")

                    if not all([v_name, shot_number is not None, position, timestamp_seconds is not None]):
                        continue

                    # Initialize video data if not exists
                    if v_name not in shots_timing_data:
                        shots_timing_data[v_name] = {}

                    # Initialize shot data if not exists
                    if shot_number not in shots_timing_data[v_name]:
                        shots_timing_data[v_name][shot_number] = {
                            "start_time": None,
                            "end_time": None,
                            "timestamps": {}
                        }

                    # Store timestamp for this position
                    shots_timing_data[v_name][shot_number]["timestamps"][position] = timestamp_seconds

                total_processed += len(points)
                if total_processed % 1000 == 0:
                    print(f"   Processed {total_processed} points...")

                if not next_page:
                    break

            # Step 2: Calculate start_time and end_time for each shot
            print("🧮 Phase 2: Calculating shot timing boundaries...")
            shot_timing_map = {}  # {video_name: {shot_number: {start_time, end_time}}}

            for v_name, shots in shots_timing_data.items():
                shot_timing_map[v_name] = {}

                for shot_number, shot_data in shots.items():
                    timestamps = shot_data["timestamps"]

                    # Get start and end timestamps
                    start_time = timestamps.get("start")
                    end_time = timestamps.get("end")

                    # If start or end is missing, use available positions
                    if start_time is None or end_time is None:
                        all_timestamps = list(timestamps.values())
                        if all_timestamps:
                            start_time = start_time or min(all_timestamps)
                            end_time = end_time or max(all_timestamps)

                    if start_time is not None and end_time is not None:
                        shot_timing_map[v_name][shot_number] = {
                            "start_time": float(start_time),
                            "end_time": float(end_time)
                        }

            print(f"   Found timing data for {sum(len(shots) for shots in shot_timing_map.values())} shots")

            # Step 3: Update all points with start_time and end_time
            print("✏️  Phase 3: Updating point payloads...")
            updated_count = 0

            for v_name, shots in shot_timing_map.items():
                for shot_number, timing in shots.items():
                    try:
                        # Update all points for this video and shot
                        update_filter = Filter(
                            must=[
                                FieldCondition(key="video.name", match=MatchValue(value=v_name)),
                                FieldCondition(key="shot.number", match=MatchValue(value=shot_number))
                            ]
                        )

                        self.client.set_payload(
                            collection_name=self.collection_name,
                            payload={
                                "start_time": timing["start_time"],
                                "end_time": timing["end_time"]
                            },
                            points=update_filter
                        )

                        updated_count += 1
                        if updated_count % 50 == 0:
                            print(f"   Updated {updated_count} shots...")

                    except Exception as e:
                        logger.error(f"Failed to update shot {v_name}_shot_{shot_number}: {e}")
                        continue

            print(f"✅ Successfully updated {updated_count} shots with start_time and end_time fields")
            return True

        except Exception as e:
            logger.error(f"update_shot_time_fields failed: {e}")
            return False

    def verify_shot_time_fields(self, video_name: Optional[str] = None, sample_size: int = 10):
        """
        Verify that start_time and end_time fields have been added correctly.

        Args:
            video_name: Optional video name to filter by
            sample_size: Number of sample points to check
        """
        try:
            filter_condition = None
            if video_name:
                filter_condition = Filter(
                    must=[FieldCondition(key="video.name", match=MatchValue(value=video_name))]
                )

            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_condition,
                with_payload=True,
                with_vectors=False,
                limit=sample_size
            )

            print(f"🔍 Verifying {len(points)} sample points...")

            verified_count = 0
            missing_fields = 0

            for point in points:
                payload = point.payload or {}
                video = payload.get("video") or {}
                shot = payload.get("shot") or {}
                frame = payload.get("frame") or {}

                v_name = video.get("name")
                shot_number = shot.get("number")
                position = shot.get("position")
                timestamp_seconds = frame.get("timestamp_seconds")
                start_time = payload.get("start_time")
                end_time = payload.get("end_time")

                print(f"   Point: {v_name}_shot_{shot_number}_{position}")
                print(f"     timestamp_seconds: {timestamp_seconds}")
                print(f"     start_time: {start_time}")
                print(f"     end_time: {end_time}")

                if start_time is not None and end_time is not None:
                    verified_count += 1
                    # Basic validation
                    if start_time <= timestamp_seconds <= end_time:
                        print(f"     ✅ Valid: {start_time} <= {timestamp_seconds} <= {end_time}")
                    else:
                        print(f"     ⚠️  Warning: timestamp outside shot range")
                else:
                    missing_fields += 1
                    print(f"     ❌ Missing start_time or end_time")

                print()

            print(f"📈 Verification Results:")
            print(f"   Total checked: {len(points)}")
            print(f"   With time fields: {verified_count}")
            print(f"   Missing fields: {missing_fields}")
            print(f"   Success rate: {verified_count/len(points)*100:.1f}%")

            return verified_count, missing_fields

        except Exception as e:
            logger.error(f"verify_shot_time_fields failed: {e}")
            return 0, 0

    def clear_cache(self):
        """Clear search cache"""
        self._search_cache.clear()

    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=False)



# Singleton pattern with improved initialization
_qdrant_service = None


def get_qdrant_service(
    api_key: Optional[str] = None,
    collection_name: str = "test-aic",
    vector_dim: int = 512,
    qdrant_url: Optional[str] = None,
    named_vectors: Optional[Dict[str, int]] = None
) -> QdrantService:
    """Get singleton QdrantService instance"""
    global _qdrant_service
    if _qdrant_service is None or _qdrant_service.collection_name != collection_name:
        try:
            _qdrant_service = QdrantService(
                api_key=api_key,
                collection_name=collection_name,
                vector_dim=vector_dim,
                qdrant_url=qdrant_url,
                named_vectors=named_vectors
            )
            _qdrant_service.setup_collection()
        except Exception as e:
            # Create a minimal service that won't crash
            _qdrant_service = QdrantService(collection_name=collection_name, vector_dim=vector_dim)

    return _qdrant_service


if __name__ == "__main__":
    # Test the service
    print("🧪 Testing QdrantService...")

    service = get_qdrant_service()
    health = service.health_check()
    print(f"Health: {health}")

    # Test search with random vector
    import numpy as np

    test_vector = np.random.random(512).tolist()

    start = time.time()
    results = service.search(test_vector, limit=5)
    search_time = time.time() - start

    print(f"Search test: {len(results)} results in {search_time:.4f}s")
    print(f"Service info: {service.get_collection_info()}")
