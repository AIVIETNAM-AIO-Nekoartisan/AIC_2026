# model/GPT4o_service.py
import os
import json
from typing import List
from openai import OpenAI
from dotenv import load_dotenv
import warnings
import re
from typing import List, Dict, Any

warnings.filterwarnings("ignore")

class GPT4oService:
    def __init__(self, model_name: str = None, use_json_mode: bool = True):
        """
        Khởi tạo LLM service (provider-agnostic).

        Provider chọn qua env LLM_PROVIDER = "deepseek" (mặc định) | "openai".
        DeepSeek dùng API tương thích OpenAI nên tái dùng nguyên SDK + prompt.
        Nếu provider chính lỗi/timeout, tự fallback sang provider còn lại
        (miễn là key của nó có trong .env).

        Args:
            model_name (str): ép model cho provider chính (mặc định lấy theo env).
            use_json_mode (bool): dùng JSON response format.
        """
        load_dotenv()
        self.use_json_mode = use_json_mode
        self.provider = os.getenv("LLM_PROVIDER", "deepseek").strip().lower()
        self._model_override = model_name
        # Chuỗi ưu tiên: [(client, model), ...] — phần tử đầu là primary.
        self._chain = []
        self._initialize_clients()

    def _make_client(self, provider: str):
        """Trả về (OpenAI client, model_name) cho provider, hoặc None nếu thiếu key."""
        # timeout bắt buộc: SDK mặc định 600s + retry -> một request treo
        # là UI đơ vài phút (mạng tới LLM ở VN vốn chậm ~5-10s/call)
        if provider == "deepseek":
            key = os.getenv("DEEPSEEK_API_KEY")
            if not key:
                return None
            base = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
            model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
            client = OpenAI(api_key=key, base_url=base, timeout=30.0, max_retries=1)
            return (client, model)
        if provider == "openai":
            key = os.getenv("OPENAI_API_KEY")
            if not key:
                return None
            model = os.getenv("OPENAI_MODEL", "gpt-4o")
            client = OpenAI(api_key=key, timeout=30.0, max_retries=1)
            return (client, model)
        return None

    def _initialize_clients(self):
        """Dựng chuỗi primary + fallback theo LLM_PROVIDER."""
        order = (
            ["deepseek", "openai"]
            if self.provider == "deepseek"
            else ["openai", "deepseek"]
        )
        for prov in order:
            made = self._make_client(prov)
            if made is None:
                continue
            client, model = made
            # cho phép ép model cho đúng provider chính
            if prov == order[0] and self._model_override:
                model = self._model_override
            self._chain.append((prov, client, model))

        if not self._chain:
            raise ValueError(
                "Chưa thiết lập key cho LLM nào — cần DEEPSEEK_API_KEY "
                "(provider=deepseek) hoặc OPENAI_API_KEY trong file .env"
            )
        # Back-compat: nhiều nơi tham chiếu self.client / self.model_name
        self.client = self._chain[0][1]
        self.model_name = self._chain[0][2]
        chain_desc = " -> ".join(f"{p}:{m}" for p, _, m in self._chain)
        print(f"LLM service khởi tạo thành công. Chuỗi provider: {chain_desc}")

    def _chat(self, **params):
        """
        Gọi chat.completions.create với chuỗi provider primary -> fallback.
        Model được set theo từng provider (override 'model' trong params).
        """
        last_err = None
        for prov, client, model in self._chain:
            params["model"] = model
            try:
                return client.chat.completions.create(**params)
            except Exception as e:  # noqa: BLE001 — thử fallback
                last_err = e
                print(f"⚠️ LLM provider '{prov}' lỗi ({e}); thử fallback...")
        raise last_err

    def expand_query(self, original_query: str, num_expansions: int = 5) -> List[str]:
        """
        Sử dụng GPT-4o để mở rộng một truy vấn tìm kiếm gốc.

        Args:
            original_query (str): Truy vấn tìm kiếm ban đầu của người dùng.
            num_expansions (int): Số lượng truy vấn mở rộng mong muốn.

        Returns:
            List[str]: Danh sách các truy vấn đã được mở rộng.
        """
        print(f"🚀 Đang mở rộng truy vấn: '{original_query}'...")

        system_prompt = f"""You are an expert in search query optimization for a visual and video retrieval system. Your task is to take a user's query in any language and expand it into exactly {num_expansions} diverse, visually descriptive English queries.

Rules:

1. **Include Direct Translation First:** The VERY FIRST query must be the simple, direct English translation of the original query.

2. **Stay True to Original:** Don't add objects, actions, or details not present in the original query. Only vary the visual perspective, setting, or description style.

3. **Visual Descriptions:** Create short, descriptive captions focusing on different visual aspects of the same core subject/action.

4. **Avoid Abstract Concepts:** No categorical lists or generic concepts.

Return as JSON with key "queries" containing the list of query strings.

### Example:
User query: "gato jugando" (Spanish)
Your JSON response:
{{
  "queries": [
    "cat playing",
    "a playful cat in motion",
    "cat playing indoors",
    "close-up of a cat playing",
    "energetic cat playing actively"
  ]
}}
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f'User query: "{original_query}"\nYour JSON response:'}
        ]

        try:
            # Cấu hình request
            request_params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 300,
            }

            # Thêm JSON mode nếu được bật
            if self.use_json_mode:
                request_params["response_format"] = {"type": "json_object"}

            response = self._chat(**request_params)
            response_content = response.choices[0].message.content

            # Parse JSON từ response
            try:
                data = json.loads(response_content)
                expanded_queries = data.get("queries", [])
                if expanded_queries:
                    print(f"✅ Mở rộng thành công! Tạo được {len(expanded_queries)} truy vấn")
                    print("Các truy vấn mở rộng:")
                    for i, query in enumerate(expanded_queries, 1):
                        print(f"  {i}. {query}")
                    return expanded_queries[:num_expansions]
            except json.JSONDecodeError as json_error:
                print(f"❌ Lỗi parse JSON: {json_error}")

            print("⚠️ Không thể parse JSON, trả về truy vấn gốc")
            return [original_query]

        except Exception as e:
            print(f"❌ Lỗi khi mở rộng truy vấn: {e}")
            return [original_query]

    def generate_text(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
        """
        Tạo văn bản sử dụng GPT-4o.

        Args:
            prompt (str): Input prompt
            max_tokens (int): Số token tối đa trong response
            temperature (float): Độ sáng tạo (0.0 - 1.0)

        Returns:
            str: Generated text
        """
        try:
            response = self._chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            generated_text = response.choices[0].message.content
            print(f"✅ Tạo văn bản thành công!")
            return generated_text

        except Exception as e:
            print(f"❌ Lỗi khi tạo văn bản: {e}")
            return ""

    def check_api_status(self) -> bool:
        """
        Kiểm tra trạng thái API key và kết nối.

        Returns:
            bool: True nếu API hoạt động bình thường
        """
        try:
            response = self._chat(
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10,
                temperature=0
            )
            print("✅ API key hoạt động bình thường")
            return True
        except Exception as e:
            print(f"❌ Lỗi kết nối API: {e}")
            return False

    def get_model_info(self) -> dict:
        """
        Lấy thông tin về model đang sử dụng.

        Returns:
            dict: Thông tin model
        """
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "chain": [f"{p}:{m}" for p, _, m in self._chain],
            "use_json_mode": self.use_json_mode,
            "api_key_configured": bool(self._chain),
        }

    def extract_temporal_events(self, query: str, max_events: int = 6) -> List[str]:
        """
        Tách truy vấn thành chuỗi sự kiện theo thứ tự thời gian.
        Trả về List[str] các sự kiện bằng tiếng Anh, bám sát nội dung câu gốc.
        - Không thêm/bịa thông tin không có trong câu gốc.
        - Mỗi sự kiện ngắn gọn (≤ 12 từ), hiện tại đơn/miêu tả gọn.
        - Nếu chỉ có 1 sự kiện thì trả về list 1 phần tử.
        - Không có fallback: nếu lỗi/không parse được JSON -> raise ValueError.
        """
        if not query or not query.strip():
            raise ValueError("Empty query")

        system_prompt = (
            "You are a temporal event extractor for video retrieval.\n"
    f"Split the user's query into at most {max_events} atomic events in strict chronological order.\n"
    "Do NOT invent details or events not stated or strongly implied by the query.\n"
    "Write each event in concise English (≤12 words), present tense or imperative.\n"

    "Merge co-occurring visual attributes into the SAME event "
    "(e.g., colors, object details, prepositional phrases like 'with/on/in/at', "
    "Vietnamese phrases 'với', 'có', 'ở'). "
    "Do NOT split a scene and its attributes into separate events.\n"
    "If the query has N temporal connectors (e.g., 'đầu tiên/tiếp theo/sau đó/cuối cùng', "
    "'first/then/next/finally'), usually return N+1 events, unless the text clearly adds/removes steps.\n"
    'Return strict JSON: {"events": ["event 1", "event 2", ...]} with no extra text.\n'

    "Example:\n"
    'VN: "Có vài người đội mũ xanh, sau đó là cảnh bản đồ quy hoạch, '
    'tiếp theo là cảnh con kênh cạnh con đường, ở trên kênh có tàu màu đỏ"\n'
    'EN events:\n'
    '["Several people wear green hats", "Show planning map", '
    '"Scene of canal beside road with a red boat"]\n'
        )
        user_prompt = f'User query:\n"""{query}"""\nReturn JSON now:'

        request_params = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 500,
        }
        if self.use_json_mode:
            request_params["response_format"] = {"type": "json_object"}

        resp = self._chat(**request_params)
        content = resp.choices[0].message.content

        try:
            data = json.loads(content)
        except Exception as e:
            raise ValueError(f"Model did not return valid JSON: {e}")

        events = data.get("events", [])
        if not isinstance(events, list):
            raise ValueError("JSON must contain 'events' as a list")

        # Làm sạch & cắt theo max_events
        cleaned = [str(s).strip() for s in events if str(s).strip()]
        if not cleaned:
            raise ValueError("No events extracted from model response")
        print(cleaned[:max_events])

        return cleaned[:max_events]

    def analyze_intelligent_query(self, query: str) -> Dict[str, Any]:
        """
        Phân tích query phức tạp và đề xuất strategy tìm kiếm đa phương thức.
        Sử dụng logic suy luận chi tiết, tách biệt rõ ràng KIS (visual), OCR (text) và ASR (spoken keywords).
        """
        if not query or not query.strip():
            raise ValueError("Empty query")

        system_prompt = """You are an expert Query Strategist for a multimedia retrieval system.
    Your task is to parse a user's query into a precise, weighted, multimodal search plan.
    You must differentiate between:
    1.  *Visual Concepts* (What you SEE: actions, objects, scenes) → KIS
    2.  *On-Screen Text* (What you READ: banners, names, numbers) → OCR
    3.  *Spoken Keywords* (What you HEAR: topics, lyrics, speech) → ASR

    --- COMPONENT & LANGUAGE RULES ---

    1.  *KIS (Visual Concepts):*
        * Describes only the visual elements: actions, scenes, objects, colors, settings.
        * ALWAYS translate to English for the SigLIP embedding model.
        * Example: "cô gái hát" → "woman singing", "người phát biểu" → "man speaking at podium".

    2.  *OCR (On-Screen Text):*
        * Contains only the exact text, numbers, or names likely visible on-screen (e.g., banners, logos, subtitles, scores, jersey names).
        * Keep ORIGINAL language for Elasticsearch exact matching.
        * Example: "Ronaldo" → "Cristiano Ronaldo 7", "đồng hồ 10 phút" → "10:00 10 phút".

    3.  *ASR (Spoken Keywords):*
        * Contains only the specific keywords, topics, or lyrics likely spoken in the audio.
        * NEVER include verbs describing the speech/singing (like "phát biểu", "hát bài").
        * Keep ORIGINAL language for Elasticsearch exact matching.
        * Example: "phát biểu về AI" → "AI trí tuệ nhân tạo", "hát bài Despacito" → "Despacito".

    --- STRATEGIC WEIGHTING HEURISTICS (Total must sum to 1.0) ---

    1.  *HEURISTIC A: VISUAL-DRIVEN (KIS High)*
        * Query is dominated by actions, attributes, or scenes.
        * Example: "người mặc áo đỏ chạy bộ trên bãi biển" → KIS: 0.9, OCR: 0.0, ASR: 0.1 (background noise).

    2.  *HEURISTIC B: TEXT-DRIVEN (OCR High)*
        * Query contains specific text, numbers, or scores that are precise signals.
        * Example: "video có đồng hồ hiện 10:30" → OCR: 0.7, KIS: 0.3 (visual context), ASR: 0.0.

    3.  *HEURISTIC C: AUDIO-DRIVEN (ASR High)*
        * Query is dominated by spoken topics, dialogue, or song lyrics.
        * Example: "bài phát biểu về kinh tế vĩ mô" → ASR: 0.7, KIS: 0.2 (visual is just 'man talking'), OCR: 0.1.

    4.  *HEURISTIC D: HYBRID (Balanced)*
        * This is the most common case. Distribute weight based on the query's components.
        * Example: "Thủ tướng Phạm Minh Chính (OCR/ASR) phát biểu về AI (ASR)"
            * Intent: Find a specific person (OCR) talking about a specific topic (ASR). Visual (KIS) is generic.
            * Weights: ASR: 0.5, OCR: 0.4, KIS: 0.1.
        * Example: "cô gái mặc áo đỏ (KIS) hát bài Despacito (ASR)"
            * Intent: Find a specific visual (KIS) combined with a specific lyric (ASR). OCR is irrelevant.
            * Weights: KIS: 0.5, ASR: 0.5, OCR: 0.0.
        * Example: "Cristiano Ronaldo (OCR) ghi bàn (KIS)"
            * Intent: Find a specific action (KIS) by a specific person (OCR, name on jersey). ASR (commentary) is secondary.
            * Weights: KIS: 0.6, OCR: 0.3, ASR: 0.1.

    --- EXAMPLES (Applying the new logic) ---

    Query: "Thủ tướng Phạm Minh Chính phát biểu về AI"
    - Components:
        "kis": "man speaking at podium, formal event, politician"
        "ocr": "Thủ tướng Phạm Minh Chính"
        "asr": "Thủ tướng Phạm Minh Chính trí tuệ nhân tạo AI"
    - Weights: {"kis": 0.1, "ocr": 0.4, "asr": 0.5}
    - Reasoning: "Primary intent is a spoken topic (ASR) by a specific entity (OCR). Visual (KIS) is generic."

    Query: "cô gái mặc áo đỏ hát bài Despacito"
    - Components:
        "kis": "woman wearing red dress singing, performance"
        "ocr": null
        "asr": "Despacito"
    - Weights: {"kis": 0.5, "ocr": 0.0, "asr": 0.5}
    - Reasoning: "Intent is balanced between a specific visual (KIS) and a specific song lyric (ASR)."

    Query: "Tìm cảnh nấu ăn có đồng hồ hiện 10 phút"
    - Components:
        "kis": "cooking scene in kitchen, preparing food"
        "ocr": "10:00 10 phút"
        "asr": null
    - Weights: {"kis": 0.6, "ocr": 0.4, "asr": 0.0}
    - Reasoning: "Intent is a visual scene (KIS) combined with a highly specific on-screen text (OCR)."

    --- FINAL OUTPUT ---
    Return ONLY valid JSON: {"components": {"kis": "...", "ocr": "...", "asr": "..."}, "weights": {"kis": 0.x, "ocr": 0.x, "asr": 0.x}, "reasoning": "Briefly explain the weighting logic."}"""

        user_prompt = f'Analyze this query: "{query}"\nReturn JSON now:'

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            request_params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": 0.0, # Temp 0.0 để bám sát logic
                "max_tokens": 1024,
            }

            if self.use_json_mode:
                request_params["response_format"] = {"type": "json_object"}

            response = self._chat(**request_params)
            response_content = response.choices[0].message.content

            data = json.loads(response_content)

            # --- PHẦN LOGIC IN VÀ CHUẨN HÓA CỦA BẠN (GIỮ NGUYÊN) ---
            print(f"\n🧠 GPT-4o Analysis Results (Corrected Logic):")
            print(f"📝 Original Query: {query}")

            print(f"📊 Components:")
            components = data.get("components", {})
            for comp, content in components.items():
                if content:
                    print(f"   - {comp.upper()}: '{content}'")
                else:
                    print(f"   - {comp.upper()}: null")

            weights = data.get("weights", {})
            print(f"⚖️  Weights: KIS={weights.get('kis', 0):.2f} | OCR={weights.get('ocr', 0):.2f} | ASR={weights.get('asr', 0):.2f}")
            print(f"💭 Reasoning: {data.get('reasoning', 'N/A')}")
            print("-" * 60)

            # Normalize weights if needed
            weights = data["weights"]
            total_weight = sum(weights.values())
            if total_weight > 0.01 and (total_weight < 0.95 or total_weight > 1.05):
                print(f"⚠️  Normalizing weights (Total was {total_weight:.2f})")
                for key in weights:
                    weights[key] = weights[key] / total_weight

            # Handle null components
            components = data["components"]
            for component in ["kis", "ocr", "asr"]:
                if components.get(component) is None:
                    weights[component] = 0.0
                elif weights.get(component, 0) == 0.0:
                    components[component] = None

            data.pop('context_type', None)

            return data

        except Exception as e:
            print(f"❌ Lỗi: {e}")
            # Fallback to KIS-only
            return {
                "components": {"kis": query, "ocr": None, "asr": None},
                "weights": {"kis": 1.0, "ocr": 0.0, "asr": 0.0},
                "reasoning": f"Fallback due to analysis error: {str(e)}"
            }


# --- Ví dụ cách sử dụng ---
if __name__ == "__main__":
    # try:
    #     # Khởi tạo service
    #     gpt4o_service = GPT4oService(use_json_mode=True)

    #     # Kiểm tra API status
    #     if not gpt4o_service.check_api_status():
    #         print("❌ Không thể kết nối tới OpenAI API")
    #         exit(1)

    #     # Test query expansion
    #     my_query = "cô gái đang nấu ăn trong bếp"
    #     expanded_list = gpt4o_service.expand_query(my_query)

    #     print("KẾT QUẢ MỞ RỘNG TRUY VẤN")
    #     print(f"Truy vấn gốc: {my_query}")
    #     print("Các truy vấn mở rộng:")
    #     for i, query in enumerate(expanded_list, 1):
    #         print(f"  {i}. {query}")

    #     print("\n" + "="*30 + "\n")

    #     # Test với query khác
    #     another_query = "tai nạn giao thông trên đường cao tốc"
    #     expanded_list_2 = gpt4o_service.expand_query(another_query, num_expansions=4)
    #     print("KẾT QUẢ MỞ RỘNG TRUY VẤN")
    #     print(f"Truy vấn gốc: {another_query}")
    #     print("Các truy vấn mở rộng:")
    #     for i, query in enumerate(expanded_list_2, 1):
    #         print(f"  {i}. {query}")

    #     print("\n" + "="*30 + "\n")

    #     # Test generate text
    #     test_prompt = "What is 2x + 3 = 7? Solve for x."
    #     response = gpt4o_service.generate_text(test_prompt)
    #     print("Math problem response:")
    #     print(response)

    #     print("\n" + "="*30 + "\n")

    #     # Hiển thị thông tin model
    #     model_info = gpt4o_service.get_model_info()
    #     print("THÔNG TIN MODEL:")
    #     for key, value in model_info.items():
    #         print(f"  {key}: {value}")

    # except ValueError as e:
    #     print(f"❌ Lỗi khởi tạo: {e}")
    # except Exception as e:
    #     print(f"❌ Lỗi: {e}")
    gpt4o_service = GPT4oService(use_json_mode=True)
    q = "Đầu tiên mở cổng, sau đó xe tải đi vào, rồi nhân viên kiểm tra hàng, cuối cùng ký biên bản"
    events = gpt4o_service.extract_temporal_events(q, max_events=6)
    print("EVENTS:", events)
