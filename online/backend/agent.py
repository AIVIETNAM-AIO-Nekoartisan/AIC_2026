"""Chế độ Agent: GPT-4o cầm các tool search sẵn có, tự chạy nhiều vòng, hỏi lại
người dùng khi thiếu thông tin, và trả về tập keyframe ứng viên để người dùng chọn nộp.

Stateless: frontend giữ lịch sử hội thoại và gửi lại mỗi lượt.
"""
import json
import logging
from typing import Any, Dict, List, Optional

from online.backend.resources import resources
from online.backend.search_kis import process_kis_search

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 6      # số vòng gọi tool tối đa trong 1 lượt, tránh loop vô hạn
MAX_RESULTS_TO_MODEL = 12  # số kết quả tóm tắt cho model đọc (tiết kiệm token)

SYSTEM_PROMPT = """Bạn là trợ lý tìm kiếm video cho cuộc thi AI Challenge (dạng Video Browser Showdown).
Người dùng mô tả một cảnh trong kho video tin tức tiếng Việt; nhiệm vụ của bạn là tìm ra đúng keyframe đó.

Công cụ:
- search_kis: tìm bằng ngữ nghĩa hình ảnh (mô tả nên viết bằng TIẾNG ANH, ngắn, tả thị giác).
  Có thể lọc theo object (vd [{"cls":"person","count":2}]) và giới hạn 1 video.
- search_asr: tìm theo lời nói trong video (tiếng Việt).
- search_ocr: tìm theo chữ hiện trên màn hình (tiếng Việt).
- subagent_summarize: giao cho trợ lý phụ đọc toàn bộ caption của kết quả hiện tại và tổng hợp
  thành các nhóm cảnh + gợi ý nên hỏi gì.
- ask_user: hỏi lại người dùng khi mô tả còn mơ hồ hoặc khi cần họ xác nhận hướng đi.

Cách làm việc (vòng lặp thu hẹp dần):
1. Chạy search trước bằng thông tin đang có — đừng hỏi khi chưa thử tìm.
2. Đọc kết quả: bạn thấy video, timestamp, object phát hiện được và độ phân tán theo video.
2b. Gọi subagent_summarize để trợ lý phụ đọc caption đầy đủ và chỉ ra các nhóm cảnh — làm việc này
   TRƯỚC khi hỏi, vì bạn chỉ thấy tóm tắt ngắn còn trợ lý phụ đọc được toàn bộ mô tả chi tiết.
3. LUÔN kết thúc lượt bằng ask_user để thu hẹp tiếp — dựa vào phần "NÊN HỎI" của trợ lý phụ.
   Một câu NGẮN, CỤ THỂ, kèm 2-4 phương án
   bấm nhanh. Ưu tiên hỏi thứ giúp cắt bớt nhiều nhất:
   - bối cảnh (trong trường quay / ngoài đường / trong phòng họp…)
   - số người hoặc vật thể chính nhìn thấy
   - chữ trên màn hình hoặc lời đang nói
   - nếu kết quả đã dồn về vài video: hỏi người dùng ảnh nào giống nhất / có phải video X không
4. Nhận câu trả lời -> tìm lại chặt hơn: thêm object_filters, đặt video_filter, đổi mô tả tiếng Anh.
5. Chỉ dừng hỏi khi người dùng nói đã tìm thấy hoặc bảo dừng. Khi đó tóm tắt ngắn và nhắc họ bấm
   vào ảnh để xem video, kiểm tra rồi nộp.

Không bịa nội dung ảnh mà bạn không thấy trong dữ liệu tool trả về.
Luôn nói với người dùng bằng tiếng Việt, ngắn gọn (1-2 câu)."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_kis",
            "description": "Tìm keyframe theo mô tả thị giác (ngữ nghĩa hình ảnh). Mô tả nên bằng tiếng Anh.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Mô tả cảnh, tiếng Anh, ngắn gọn, tả thị giác"},
                    "num_results": {"type": "integer", "description": "Số kết quả (mặc định 40)"},
                    "video_filter": {"type": "string", "description": "Chỉ tìm trong 1 video, vd L27_V003"},
                    "object_filters": {
                        "type": "array",
                        "description": "Lọc theo object YOLO, vd [{\"cls\":\"person\",\"count\":2}]",
                        "items": {
                            "type": "object",
                            "properties": {
                                "cls": {"type": "string", "description": "tên class tiếng Anh, thường; rỗng = tổng số object"},
                                "count": {"type": "integer", "description": "số lượng mong muốn; bỏ trống = chỉ cần có mặt"},
                            },
                        },
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_asr",
            "description": "Tìm đoạn video theo LỜI NÓI (tiếng Việt) trong audio.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Từ/cụm từ tiếng Việt được nói"},
                    "num_results": {"type": "integer"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_ocr",
            "description": "Tìm keyframe theo CHỮ hiện trên màn hình (tiếng Việt).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Chữ xuất hiện trên hình"},
                    "num_results": {"type": "integer"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "subagent_summarize",
            "description": ("Giao cho trợ lý phụ đọc TOÀN BỘ caption của các kết quả vừa tìm được và "
                            "tổng hợp lại: có mấy nhóm cảnh khác nhau, điểm chung/khác biệt, và nên hỏi "
                            "người dùng điều gì để tách các nhóm đó. Dùng khi kết quả còn tản mát, "
                            "TRƯỚC khi gọi ask_user."),
            "parameters": {
                "type": "object",
                "properties": {
                    "focus": {"type": "string",
                              "description": "Điều người dùng đang tìm, để trợ lý phụ biết cần phân biệt theo tiêu chí nào"},
                },
                "required": ["focus"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_user",
            "description": "Hỏi người dùng một câu ngắn để thu hẹp phạm vi tìm kiếm.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "Câu hỏi ngắn, cụ thể, bằng tiếng Việt"},
                    "suggestions": {
                        "type": "array", "items": {"type": "string"},
                        "description": "2-4 phương án trả lời nhanh để người dùng bấm",
                    },
                },
                "required": ["question"],
            },
        },
    },
]


def _summarize_for_model(results: List[Dict[str, Any]]) -> str:
    """Rút gọn kết quả thành text ngắn cho model đọc (không gửi payload đầy đủ)."""
    if not results:
        return "Không có kết quả nào."
    # Phân bố theo video giúp model biết kết quả đã hội tụ hay còn tản mát
    dist: Dict[str, int] = {}
    for r in results:
        v = ((r.get("payload") or {}).get("video") or {}).get("name") or r.get("video_name") or "?"
        dist[v] = dist.get(v, 0) + 1
    top_videos = sorted(dist.items(), key=lambda kv: -kv[1])[:6]
    dist_txt = ", ".join(f"{v} ({n})" for v, n in top_videos)

    lines = []
    for i, r in enumerate(results[:MAX_RESULTS_TO_MODEL], 1):
        p = r.get("payload") or {}
        od = p.get("object_detection") or {}
        objs = ", ".join(f"{c}:{n}" for c, n in list((od.get("class_counts") or {}).items())[:5])
        ts = (p.get("frame") or {}).get("timestamp_seconds")
        cap = (p.get("caption") or "")[:90]
        lines.append(
            f"{i}. {(p.get('video') or {}).get('name', '?')} @ {ts}s"
            f" | score {r.get('score', 0):.3f}"
            + (f" | objects: {objs}" if objs else "")
            + (f" | caption: {cap}…" if cap else "")
            + (f" | text: {r.get('text', '')[:80]}" if r.get("text") else "")
        )
    return (f"{len(results)} kết quả, trải trên {len(dist)} video: {dist_txt}\n"
            f"Top {min(len(results), MAX_RESULTS_TO_MODEL)}:\n" + "\n".join(lines))


SUBAGENT_PROMPT = """Bạn là trợ lý phụ. Bạn nhận caption (do BLIP2 sinh) của các keyframe mà hệ thống vừa tìm được,
và nhiệm vụ DUY NHẤT là tổng hợp chúng thành thông tin giúp trợ lý chính đặt câu hỏi đúng chỗ.

Trả lời đúng cấu trúc sau, ngắn gọn, bằng tiếng Việt:
NHÓM CẢNH: liệt kê 2-4 nhóm cảnh khác biệt nhau, mỗi nhóm 1 dòng, ghi rõ video nào thuộc nhóm đó.
ĐIỂM CHUNG: những gì hầu hết kết quả đều có (thông tin này KHÔNG giúp phân biệt).
KHÁC BIỆT: các tiêu chí thực sự tách được các nhóm (bối cảnh, số người, vật thể, màu sắc, trong/ngoài trời...).
NÊN HỎI: đúng MỘT câu hỏi ngắn cho người dùng, kèm 2-4 phương án tương ứng với các nhóm ở trên.

Lưu ý: caption do máy sinh nên có thể sai, nhất là phần chữ trên màn hình — đừng dựa vào chi tiết chữ."""


def _subagent_summarize(focus: str, results: List[Dict[str, Any]]) -> str:
    """Trợ lý phụ: đọc caption của kết quả hiện tại -> tổng hợp thành context cho agent chính."""
    if not results:
        return "Chưa có kết quả nào để tổng hợp — hãy chạy search trước."
    lines = []
    for r in results[:40]:
        p = r.get("payload") or {}
        cap = p.get("caption")
        if not cap:
            continue
        v = (p.get("video") or {}).get("name", "?")
        ts = (p.get("frame") or {}).get("timestamp_seconds")
        lines.append(f"- [{v} @ {ts}s] {cap}")
    if not lines:
        return "Các kết quả này chưa có caption (chưa chạy offline/ingest_captions.py)."

    try:
        resp = resources.gpt_service.client.chat.completions.create(
            model=resources.gpt_service.model_name,
            messages=[
                {"role": "system", "content": SUBAGENT_PROMPT},
                {"role": "user", "content": f"Người dùng đang tìm: {focus}\n\nCaption của {len(lines)} kết quả:\n"
                                            + "\n".join(lines)},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        logger.warning(f"subagent lỗi: {e}")
        return f"Trợ lý phụ lỗi: {e}"


def _run_tool(name: str, args: Dict[str, Any], ctx: Dict[str, Any]) -> tuple[str, List[Dict[str, Any]]]:
    """Chạy 1 tool -> (text cho model, danh sách kết quả cho UI)."""
    if name == "subagent_summarize":
        return _subagent_summarize(args.get("focus", ""), ctx.get("results") or []), []
    if name == "search_kis":
        results = process_kis_search(
            text_query=args.get("query"),
            image=None,
            num_results=int(args.get("num_results") or 40),
            score_threshold=0.0,
            query_weight=1.0,
            auto_expand=False,
            num_expansions=4,
            use_rerank=False,
            video_filter=args.get("video_filter") or None,
            object_filters=args.get("object_filters") or None,
            object_count_tolerance=3,
        )
        return _summarize_for_model(results), results

    if name in ("search_asr", "search_ocr"):
        svc = resources.es_audio if name == "search_asr" else resources.es_ocr
        if svc is None:
            return f"{name} không khả dụng (Elasticsearch chưa bật).", []
        try:
            hits = svc.search(args.get("query", ""), limit=int(args.get("num_results") or 20)) or []
        except Exception as e:
            return f"{name} lỗi: {e}", []
        if not hits:
            return ("Không tìm thấy. Lưu ý: index này có thể chưa được nạp dữ liệu "
                    "— nếu vậy hãy dùng search_kis thay thế."), []
        return _summarize_for_model(hits), hits

    if name == "ask_user":
        return "", []

    return f"Tool không tồn tại: {name}", []


def run_agent_turn(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Chạy 1 lượt: model gọi tool cho tới khi cần hỏi người dùng hoặc chốt câu trả lời."""
    if resources.gpt_service is None:
        raise RuntimeError("GPT service chưa sẵn sàng (thiếu OPENAI_API_KEY?)")

    client = resources.gpt_service.client
    model = resources.gpt_service.model_name
    convo: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}] + list(messages)

    ui_results: List[Dict[str, Any]] = []
    actions: List[Dict[str, Any]] = []   # nhật ký hành động để hiện trên UI
    question: Optional[Dict[str, Any]] = None

    for _ in range(MAX_TOOL_ROUNDS):
        resp = client.chat.completions.create(
            model=model, messages=convo, tools=TOOLS, tool_choice="auto", temperature=0.2,
        )
        msg = resp.choices[0].message
        convo.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            return {"message": msg.content or "", "messages": convo[1:], "results": ui_results,
                    "actions": actions, "question": None, "done": True}

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            if name == "ask_user":
                question = {"question": args.get("question", ""), "suggestions": args.get("suggestions") or []}
                actions.append({"tool": name, "args": args, "summary": args.get("question", "")})
                convo.append({"role": "tool", "tool_call_id": tc.id, "content": "Đã hỏi người dùng, chờ trả lời."})
                return {"message": question["question"], "messages": convo[1:], "results": ui_results,
                        "actions": actions, "question": question, "done": False}

            text, results = _run_tool(name, args, {"results": ui_results})
            if results:
                ui_results = results  # kết quả mới nhất là thứ hiển thị
            actions.append({
                "tool": name,
                "args": args,
                "summary": f"{len(results)} kết quả" if results else (text[:120] if text else ""),
            })
            convo.append({"role": "tool", "tool_call_id": tc.id, "content": text or "(không có dữ liệu)"})

    return {"message": "Đã chạy nhiều bước tìm kiếm. Bạn xem kết quả bên phải và cho tôi thêm gợi ý nhé.",
            "messages": convo[1:], "results": ui_results, "actions": actions, "question": None, "done": True}
