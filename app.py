import html
import math
from datetime import datetime
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from src.database import (
    init_db,
    insert_raw_ticket,
    update_ticket_prediction,
    get_sorted_tickets,
    delete_tickets,
    clear_all_tickets,
    score_to_priority,
)
from src.predictor import HelpdeskPredictor

st.set_page_config(
    page_title="Hệ thống Phân loại & Xếp hạng Hỗ trợ IT",
    page_icon="🎫",
    layout="wide",
)

init_db()

PAGE_SIZE = 10  # Cố định 10 bản ghi mỗi trang

if "table_version" not in st.session_state:
    st.session_state["table_version"] = 0
if "last_viewed_cell" not in st.session_state:
    st.session_state["last_viewed_cell"] = None
if "current_page" not in st.session_state:
    st.session_state["current_page"] = 1
if "last_filter_state" not in st.session_state:
    st.session_state["last_filter_state"] = None


@st.cache_resource
def load_predictor():
    return HelpdeskPredictor(model_dir="saved_model")


predictor = load_predictor()

PRIORITY_ORDER = ["critical", "high", "medium", "low", "very_low"]
PRIORITY_LEVEL = {
    "very_low": 1,
    "low": 2,
    "medium": 3,
    "high": 4,
    "critical": 5,
}
PRIORITY_BADGE = {
    "critical": "🔴 CRITICAL (Khẩn cấp)",
    "high": "🟠 HIGH (Cao)",
    "medium": "🟡 MEDIUM (Trung bình)",
    "low": "🟢 LOW (Thấp)",
    "very_low": "⚪ VERY LOW (Rất thấp)",
}


def make_auto_subject(body: str, max_len: int = 55) -> str:
    """Tự động tạo tiêu đề tóm tắt từ nội dung khi người dùng bỏ trống tiêu đề"""
    clean_line = " ".join(body.strip().split())
    if len(clean_line) <= max_len:
        return clean_line
    return clean_line[:max_len].rsplit(" ", 1)[0] + "..."


def format_priority_with_trend(effective_prio: str, base_prio: str) -> str:
    """Hiển thị mức ưu tiên kèm mũi tên ⬆️ nếu được đôn hạng do thời gian chờ phản hồi"""
    eff_label = PRIORITY_BADGE.get(effective_prio, effective_prio)
    if not base_prio or effective_prio == base_prio:
        return eff_label

    eff_lv = PRIORITY_LEVEL.get(effective_prio, 0)
    base_lv = PRIORITY_LEVEL.get(base_prio, 0)
    if eff_lv > base_lv:
        return f"{eff_label} ⬆️ (Ban đầu: {base_prio})"
    elif eff_lv < base_lv:
        return f"{eff_label} ⬇️ (Ban đầu: {base_prio})"
    return eff_label


def format_wait_progress(row: dict, enable_wait_boost: bool = True) -> str:
    """Định dạng tiến độ thời gian chờ phản hồi theo giây"""
    if not enable_wait_boost:
        return f"{int(row['wait_seconds'])}/{int(row['wait_threshold'])}s ({row['wait_status']})"
    return f"{int(row['wait_seconds'])}/{int(row['wait_threshold'])}s ({row['wait_pct']}%) {row['wait_status']}"


def parse_uploaded_file(uploaded_file) -> pd.DataFrame:
    """Đọc file CSV hoặc Excel (cho phép cột Tiêu đề bị bỏ trống hoặc chỉ có cột Nội dung)"""
    file_name = uploaded_file.name.lower()
    if file_name.endswith(".csv"):
        df_raw = None
        for enc in ("utf-8-sig", "utf-8", "cp1258", "latin1"):
            try:
                uploaded_file.seek(0)
                df_raw = pd.read_csv(uploaded_file, encoding=enc)
                break
            except Exception:
                continue
        if df_raw is None:
            raise ValueError("Không thể đọc định dạng mã hóa của tệp CSV. Hãy lưu tệp dưới dạng CSV UTF-8.")
    elif file_name.endswith((".xlsx", ".xls")):
        try:
            uploaded_file.seek(0)
            df_raw = pd.read_excel(uploaded_file)
        except ImportError:
            raise ImportError("Máy chưa cài thư viện đọc Excel (openpyxl). Vui lòng dùng tệp .CSV hoặc chạy: pip install openpyxl")
    else:
        raise ValueError("Chỉ hỗ trợ tệp định dạng .CSV hoặc .XLSX.")

    col_map = {}
    for col in df_raw.columns:
        c_norm = str(col).strip().lower()
        if c_norm in ("tiêu đề", "tieu de", "subject", "title"):
            col_map[col] = "subject"
        elif c_norm in ("nội dung", "noi dung", "body", "content", "description", "mô tả", "mo ta"):
            col_map[col] = "body"

    df_clean = df_raw.rename(columns=col_map)
    if "body" not in df_clean.columns:
        raise ValueError(
            "Tệp tải lên phải có cột 'Nội dung' (hoặc 'body'). Cột 'Tiêu đề' có thể có hoặc bỏ trống."
        )

    df_clean = df_clean.dropna(subset=["body"]).copy()
    df_clean["body"] = df_clean["body"].astype(str).str.strip()
    df_clean = df_clean[df_clean["body"] != ""].reset_index(drop=True)

    if "subject" in df_clean.columns:
        df_clean["raw_subject"] = df_clean["subject"].fillna("").astype(str).str.strip()
    else:
        df_clean["raw_subject"] = ""

    df_clean["subject"] = df_clean.apply(
        lambda r: r["raw_subject"] if r["raw_subject"] else make_auto_subject(r["body"]),
        axis=1,
    )
    return df_clean[["raw_subject", "subject", "body"]]


@st.dialog("📄 Chi tiết nội dung yêu cầu hỗ trợ", width="large")
def show_ticket_popup(ticket_row: dict, enable_wait_boost: bool = True):
    """Cửa sổ nổi cỡ chữ lớn (22px - 24px), hỗ trợ kéo thả toàn bộ khung trắng"""
    safe_subject = html.escape(str(ticket_row["subject"]))
    safe_body = html.escape(str(ticket_row["body"]))
    prio_label = format_priority_with_trend(ticket_row["effective_priority"], ticket_row["base_priority"])
    wait_info = format_wait_progress(ticket_row, enable_wait_boost=enable_wait_boost)

    st.markdown(
        f"""
        <div style="font-size: 18px; line-height: 1.75; margin-bottom: 8px;">
            <b>Hạng toàn cục:</b> <code>#{ticket_row['global_rank']}</code> &nbsp;|&nbsp;
            <b>Mã phiếu:</b> <code>#{ticket_row['id']}</code> &nbsp;|&nbsp;
            <b>Mức ưu tiên:</b> {prio_label}<br/>
            <b>Điểm ưu tiên tổng:</b> <code>{ticket_row['final_score']:.2f}</code> (Điểm nội dung: <code>{ticket_row['urgency_score']:.2f}</code> + Cộng thời gian: <code>+{ticket_row['time_bonus']:.2f}</code>) &nbsp;|&nbsp;
            <b>Thời gian chờ phản hồi:</b> <code>{wait_info}</code><br/>
            <b>Phân loại (Type):</b> <code>{ticket_row['pred_type']}</code> ({ticket_row['conf_type']:.1f}%) &nbsp;|&nbsp;
            <b>Bộ phận (Queue):</b> <code>{ticket_row['pred_queue']}</code> ({ticket_row['conf_queue']:.1f}%) &nbsp;|&nbsp;
            <b>Gửi lúc:</b> <code>{ticket_row['created_at']}</code>
        </div>
        <hr style="margin: 12px 0;"/>
        <div style="font-size: 20px; font-weight: 700; margin-bottom: 8px;">📌 Tiêu đề đầy đủ:</div>
        <div style="
            font-size: 24px;
            font-weight: 700;
            line-height: 1.45;
            padding: 14px 18px;
            border-radius: 10px;
            background: rgba(59, 130, 246, 0.12);
            border-left: 6px solid #3b82f6;
            margin-bottom: 18px;
            word-break: break-word;
        ">{safe_subject}</div>
        <div style="font-size: 20px; font-weight: 700; margin-bottom: 8px;">📝 Nội dung chi tiết đầy đủ:</div>
        <div data-scrollable="true" style="
            font-size: 22px;
            line-height: 1.65;
            padding: 16px 20px;
            border-radius: 10px;
            background: rgba(128, 128, 128, 0.10);
            border: 1px solid rgba(128, 128, 128, 0.30);
            white-space: pre-wrap;
            word-break: break-word;
            max-height: 360px;
            overflow-y: auto;
            cursor: text;
        ">{safe_body}</div>
        """,
        unsafe_allow_html=True,
    )

    components.html(
        """
        <script>
        (function() {
            const iframeEl = window.frameElement;
            if (!iframeEl) return;
            iframeEl.style.display = "none";

            const parentWin = window.parent;
            const parentDoc = parentWin.document;

            function findWhiteModalBox() {
                let el = iframeEl.parentElement;
                let candidate = null;
                while (el && el !== parentDoc.body) {
                    const style = parentWin.getComputedStyle(el);
                    const rect = el.getBoundingClientRect();
                    const bg = style.backgroundColor;
                    const hasSolidBg = bg && bg !== "transparent" && bg !== "rgba(0, 0, 0, 0)";
                    const isModalSized = rect.width > 250 && rect.width < parentWin.innerWidth * 0.98 && rect.height > 150;
                    if (hasSolidBg && isModalSized) {
                        candidate = el;
                    }
                    el = el.parentElement;
                }
                return candidate || parentDoc.querySelector('[role="dialog"]');
            }

            const modalBox = findWhiteModalBox();
            if (!modalBox || modalBox.dataset.modalDragReady === "true") return;
            modalBox.dataset.modalDragReady = "true";

            modalBox.style.cursor = "grab";
            modalBox.style.willChange = "transform";

            let offsetX = 0;
            let offsetY = 0;
            let isDragging = false;
            let startX = 0;
            let startY = 0;

            modalBox.addEventListener("mousedown", function(e) {
                if (e.button !== 0) return;
                if (e.target.closest("button, input, textarea, select, a, [data-scrollable='true']")) {
                    return;
                }
                isDragging = true;
                startX = e.clientX - offsetX;
                startY = e.clientY - offsetY;
                modalBox.style.cursor = "grabbing";
                modalBox.style.userSelect = "none";
                e.preventDefault();
            });

            const onMouseMove = function(e) {
                if (!isDragging) return;
                if (!parentDoc.body.contains(modalBox)) {
                    parentWin.removeEventListener("mousemove", onMouseMove);
                    parentWin.removeEventListener("mouseup", onMouseUp);
                    return;
                }
                offsetX = e.clientX - startX;
                offsetY = e.clientY - startY;
                modalBox.style.transform = `translate3d(${offsetX}px, ${offsetY}px, 0)`;
            };

            const onMouseUp = function() {
                if (!isDragging) return;
                isDragging = false;
                modalBox.style.cursor = "grab";
                modalBox.style.userSelect = "auto";
            };

            parentWin.addEventListener("mousemove", onMouseMove);
            parentWin.addEventListener("mouseup", onMouseUp);
        })();
        </script>
        """,
        height=0,
    )


st.title("🎫 Hệ thống Phân loại & Xếp hạng Ưu tiên Yêu cầu Hỗ trợ (Multi-Task PhoBERT)")
st.caption("Công thức xếp hạng động: Điểm ưu tiên = Điểm nội dung (AI đánh giá 1–5) + Điểm cộng theo tỷ lệ thời gian chờ phản hồi (tính theo giây)")

with st.expander("⚙️ Cấu hình Ngưỡng thời gian chờ phản hồi (giây) & Hệ số tăng tốc quá hạn", expanded=False):
    st.caption("Mặc định tối ưu cho trình chiếu nhanh (20s – 120s): chạm 100% ngưỡng thời gian chờ được cộng +1.00 điểm (đôn 1 bậc ưu tiên); vượt quá 100% sẽ cộng tăng tốc để nổi bật lên đầu.")
    s1, s2, s3, s4, s5, s6 = st.columns(6)
    with s1:
        wait_crit = st.number_input("🔴 Critical (giây)", min_value=5, max_value=86400, value=20, step=5)
    with s2:
        wait_high = st.number_input("🟠 High (giây)", min_value=5, max_value=86400, value=40, step=5)
    with s3:
        wait_med = st.number_input("🟡 Medium (giây)", min_value=5, max_value=86400, value=60, step=10)
    with s4:
        wait_low = st.number_input("🟢 Low (giây)", min_value=5, max_value=86400, value=90, step=15)
    with s5:
        wait_vlow = st.number_input("⚪ Very Low (giây)", min_value=5, max_value=86400, value=120, step=15)
    with s6:
        overdue_penalty = st.number_input(
            "Hệ số tăng tốc quá hạn",
            min_value=0.5,
            max_value=5.0,
            value=1.5,
            step=0.5,
            format="%.1f",
            help="Điểm cộng thêm cho mỗi 100% thời gian vượt quá ngưỡng chờ phản hồi.",
        )

wait_thresholds_cfg = {
    "critical": float(wait_crit),
    "high": float(wait_high),
    "medium": float(wait_med),
    "low": float(wait_low),
    "very_low": float(wait_vlow),
}

# --- PHẦN 1: NHẬP YÊU CẦU HỖ TRỢ (THỦ CÔNG HOẶC UPLOAD FILE) ---
st.subheader("1. Tiếp nhận yêu cầu hỗ trợ")

tab_manual, tab_upload = st.tabs(["✍️ Nhập trực tiếp từng phiếu", "📂 Tải lên file hàng loạt (CSV / Excel)"])

with tab_manual:
    with st.form("ticket_form", clear_on_submit=True):
        subject = st.text_input(
            "Tiêu đề yêu cầu (Có thể bỏ trống)",
            placeholder="Để trống nếu là tin nhắn thực tế — hệ thống sẽ tự trích từ nội dung...",
        )
        body = st.text_area(
            "Nội dung chi tiết *",
            height=105,
            placeholder="Nhập nội dung chi tiết (hỗ trợ cả văn phong nhắn tin viết tắt tiếng Việt)...",
        )
        submitted = st.form_submit_button("🚀 Gửi yêu cầu & Phân tích AI", type="primary")

    if submitted:
        if not body.strip():
            st.warning("Vui lòng nhập Nội dung chi tiết của yêu cầu.")
        else:
            raw_subject = subject.strip()
            body_clean = body.strip()
            display_subject = raw_subject if raw_subject else make_auto_subject(body_clean)

            ticket_id = insert_raw_ticket(display_subject, body_clean)
            with st.spinner("Đang phân tích ngữ nghĩa bằng Multi-Task PhoBERT..."):
                result = predictor.predict(raw_subject, body_clean)
            update_ticket_prediction(ticket_id, result)
            st.session_state["table_version"] += 1
            st.session_state["last_viewed_cell"] = None
            st.session_state["current_page"] = 1

            consistent_prio = score_to_priority(result["urgency_score"])
            st.success(
                f"✅ Đã lưu phiếu **#{ticket_id}** | "
                f"Ưu tiên AI: **{PRIORITY_BADGE[consistent_prio]}** (Điểm nội dung: `{result['urgency_score']}/5.0` - Tin cậy `{result['conf_priority']}%`) | "
                f"Phân loại: **{result['pred_type']}** | Bộ phận: **{result['pred_queue']}** | "
                f"Thời gian AI: `{result['inference_ms']} ms`"
            )

with tab_upload:
    st.markdown(
        "Tải lên tệp **`.csv`** hoặc **`.xlsx`** gồm cột **`Nội dung`** (bắt buộc) và cột **`Tiêu đề`** (tùy chọn, có thể để trống). "
        "Hệ thống sẽ tự động phân tích AI từng dòng và nhập toàn bộ vào bảng xếp hạng bên dưới."
    )

    uploaded_file = st.file_uploader(
        "Chọn tệp dữ liệu yêu cầu hỗ trợ (.csv, .xlsx)",
        type=["csv", "xlsx", "xls"],
        key="batch_ticket_uploader",
    )

    if uploaded_file is not None:
        try:
            df_import = parse_uploaded_file(uploaded_file)
            if df_import.empty:
                st.warning("Tệp không có dòng dữ liệu hợp lệ nào (cột Nội dung không được để trống).")
            else:
                st.caption(f"📋 Xem trước **{len(df_import)}** yêu cầu hợp lệ từ tệp `{uploaded_file.name}`:")
                st.dataframe(
                    df_import[["subject", "body"]].rename(columns={"subject": "Tiêu đề", "body": "Nội dung"}),
                    height=180,
                    use_container_width=True,
                    hide_index=True,
                )
                if st.button(
                    f"⚡ Phân tích AI & Nhập toàn bộ {len(df_import)} phiếu vào bảng kết quả",
                    type="primary",
                    key="btn_import_batch",
                ):
                    progress_bar = st.progress(0, text="Đang khởi động mô hình PhoBERT...")
                    total_rows = len(df_import)
                    for idx, row in df_import.iterrows():
                        progress_bar.progress(
                            int((idx / total_rows) * 100),
                            text=f"Đang phân tích phiếu {idx + 1}/{total_rows}: {row['subject'][:45]}...",
                        )
                        tid = insert_raw_ticket(row["subject"], row["body"])
                        res = predictor.predict(row["raw_subject"], row["body"])
                        update_ticket_prediction(tid, res)

                    progress_bar.progress(100, text="Hoàn tất phân tích toàn bộ dữ liệu!")
                    st.session_state["table_version"] += 1
                    st.session_state["last_viewed_cell"] = None
                    st.session_state["current_page"] = 1
                    st.success(f"✅ Đã phân tích AI và nhập thành công **{total_rows}** yêu cầu vào danh sách xếp hạng!")
        except Exception as e:
            st.error(f"Lỗi đọc tệp tải lên: {e}")

st.divider()

# --- PHẦN 2: BẢNG KẾT QUẢ TỰ ĐỘNG CẬP NHẬT, LỌC & PHÂN TRANG ---
head_col1, head_col2, head_col3 = st.columns([2.7, 1.2, 1.1])
with head_col1:
    st.subheader("2. Danh sách yêu cầu hỗ trợ (Xếp hạng ưu tiên từ trên xuống dưới)")
with head_col2:
    enable_wait_boost = st.toggle(
        "⏱️ Tính thời gian chờ",
        value=True,
        help="Tắt để xem thứ hạng và mức ưu tiên ban đầu của AI; bật để cộng điểm và đôn hạng theo thời gian chờ phản hồi.",
    )
with head_col3:
    auto_refresh = st.toggle(
        "🔄 Tự động làm mới (5s)",
        value=True,
        disabled=not enable_wait_boost,
    )


def render_priority_board():
    tickets = get_sorted_tickets(
        wait_thresholds=wait_thresholds_cfg,
        overdue_penalty=overdue_penalty,
        enable_wait_boost=enable_wait_boost,
    )

    for rank_idx, t in enumerate(tickets, start=1):
        t["global_rank"] = rank_idx

    total_cnt  = len(tickets)
    crit_cnt   = sum(1 for t in tickets if t["effective_priority"] == "critical")
    high_cnt   = sum(1 for t in tickets if t["effective_priority"] == "high")
    medium_cnt = sum(1 for t in tickets if t["effective_priority"] == "medium")
    low_cnt    = sum(1 for t in tickets if t["effective_priority"] in ("low", "very_low"))

    st.markdown(
        """
        <style>
        [data-testid="stMetric"] {
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
        }
        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"] {
            display: flex;
            justify-content: center;
            width: 100%;
            text-align: center;
        }
        [data-testid="stMetricLabel"] > div {
            justify-content: center;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Tổng số yêu cầu lưu trữ", f"{total_cnt}")
    k2.metric("🔴 Critical (Khẩn cấp)", f"{crit_cnt}")
    k3.metric("🟠 High (Cao)", f"{high_cnt}")
    k4.metric("🟡 Medium (Trung bình)", f"{medium_cnt}")
    k5.metric("🟢 Low / Very Low (Thấp)", f"{low_cnt}")

    if not tickets:
        st.info("Chưa có yêu cầu nào trong cơ sở dữ liệu. Hãy nhập thủ công hoặc tải lên file ở phần 1 để bắt đầu!")
        return

    # --- THANH BỘ LỌC THEO PRIORITY, TYPE, QUEUE ---
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        selected_priorities = st.multiselect(
            "Lọc theo Mức ưu tiên (Priority)",
            options=PRIORITY_ORDER,
            format_func=lambda p: PRIORITY_BADGE.get(p, p),
            placeholder="Tất cả mức ưu tiên",
            key="filter_priority",
        )
    with f_col2:
        type_options = list(dict.fromkeys(predictor.type_labels + [t["pred_type"] for t in tickets if t.get("pred_type")]))
        selected_types = st.multiselect(
            "Lọc theo Phân loại (Type)",
            options=type_options,
            placeholder="Tất cả loại yêu cầu",
            key="filter_type",
        )
    with f_col3:
        queue_options = list(dict.fromkeys(predictor.queue_labels + [t["pred_queue"] for t in tickets if t.get("pred_queue")]))
        selected_queues = st.multiselect(
            "Lọc theo Bộ phận (Queue)",
            options=queue_options,
            placeholder="Tất cả bộ phận tiếp nhận",
            key="filter_queue",
        )

    filtered_tickets = [
        t for t in tickets
        if (not selected_priorities or t["effective_priority"] in selected_priorities)
        and (not selected_types or t["pred_type"] in selected_types)
        and (not selected_queues or t["pred_queue"] in selected_queues)
    ]

    if not filtered_tickets:
        st.warning("Không có phiếu nào khớp với điều kiện lọc đã chọn. Hãy bỏ bớt tiêu chí lọc ở trên để xem lại danh sách!")
        return

    current_filter_state = (
        tuple(selected_priorities),
        tuple(selected_types),
        tuple(selected_queues),
    )
    if st.session_state["last_filter_state"] != current_filter_state:
        st.session_state["last_filter_state"] = current_filter_state
        st.session_state["current_page"] = 1

    total_filtered = len(filtered_tickets)
    total_pages = max(1, math.ceil(total_filtered / PAGE_SIZE))
    st.session_state["current_page"] = max(1, min(st.session_state["current_page"], total_pages))
    cur_page = st.session_state["current_page"]

    start_idx = (cur_page - 1) * PAGE_SIZE
    end_idx = start_idx + PAGE_SIZE
    page_tickets = filtered_tickets[start_idx:end_idx]

    now_str = datetime.now().strftime("%H:%M:%S")
    mode_badge = (
        "⏱️ **Đang tính thời gian chờ** (Xếp hạng động theo thời gian phản hồi)"
        if enable_wait_boost
        else "⏸️ **Không tính thời gian chờ** (Thứ hạng ban đầu của AI)"
    )
    st.caption(
        f"🕒 Cập nhật lúc **{now_str}** &nbsp;|&nbsp; {mode_badge} &nbsp;|&nbsp; "
        f"Hiển thị phiếu **{start_idx + 1}–{min(end_idx, total_filtered)}** / **{total_filtered}** (cố định {PAGE_SIZE} dòng/trang) &nbsp;|&nbsp; "
        "💡 Bấm vào ô **Tiêu đề** hoặc **Nội dung** để phóng to toàn văn."
    )

    def build_dataframe(rows_data: list) -> pd.DataFrame:
        df = pd.DataFrame(rows_data)

        df["Mức ưu tiên"] = df.apply(
            lambda row: format_priority_with_trend(row["effective_priority"], row["base_priority"]),
            axis=1,
        )
        df["Thời gian chờ phản hồi"] = df.apply(
            lambda r: format_wait_progress(r, enable_wait_boost=enable_wait_boost),
            axis=1,
        )
        df["Phân loại (Type)"] = df.apply(
            lambda r: f"{r['pred_type']} ({r['conf_type']:.1f}%)" if pd.notna(r["pred_type"]) else "", axis=1
        )
        df["Bộ phận (Queue)"] = df.apply(
            lambda r: f"{r['pred_queue']} ({r['conf_queue']:.1f}%)" if pd.notna(r["pred_queue"]) else "", axis=1
        )

        cols_to_show = [
            "global_rank",
            "id",
            "Mức ưu tiên",
            "final_score",
            "urgency_score",
            "time_bonus",
            "Thời gian chờ phản hồi",
            "Phân loại (Type)",
            "Bộ phận (Queue)",
            "subject",
            "body",
            "created_at",
        ]
        return df[cols_to_show].rename(
            columns={
                "global_rank": "Hạng",
                "id": "Mã phiếu",
                "final_score": "Điểm ưu tiên",
                "urgency_score": "Điểm nội dung",
                "time_bonus": "Cộng thời gian",
                "subject": "Tiêu đề",
                "body": "Nội dung",
                "created_at": "Thời gian gửi",
            }
        )

    df_page = build_dataframe(page_tickets)
    table_height = min(430, max(160, 44 + len(df_page) * 38))

    event = st.dataframe(
        df_page,
        height=table_height,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode=["multi-row", "single-cell"],
        column_config={
            "Hạng": st.column_config.NumberColumn("Hạng", width="small"),
            "Mã phiếu": st.column_config.NumberColumn("Mã phiếu", width="small"),
            "Mức ưu tiên": st.column_config.TextColumn("Mức ưu tiên", width="medium"),
            "Điểm ưu tiên": st.column_config.NumberColumn("Điểm ưu tiên", format="%.2f", width="small"),
            "Điểm nội dung": st.column_config.NumberColumn("Điểm nội dung", format="%.2f", width="small"),
            "Cộng thời gian": st.column_config.NumberColumn("Cộng thời gian", format="+%.2f", width="small"),
            "Thời gian chờ phản hồi": st.column_config.TextColumn("Thời gian chờ phản hồi", width="medium"),
            "Phân loại (Type)": st.column_config.TextColumn("Phân loại (Type)", width="medium"),
            "Bộ phận (Queue)": st.column_config.TextColumn("Bộ phận (Queue)", width="medium"),
            "Tiêu đề": st.column_config.TextColumn("Tiêu đề (Bấm để xem)", width="medium"),
            "Nội dung": st.column_config.TextColumn("Nội dung (Bấm để xem)", width="large"),
            "Thời gian gửi": st.column_config.TextColumn("Thời gian gửi", width="medium"),
        },
        key=f"tickets_table_{st.session_state['table_version']}_p{cur_page}_wt{int(enable_wait_boost)}",
    )

    # --- THANH ĐIỀU HƯỚNG PHÂN TRANG NHỎ GỌN Ở CHÍNH GIỮA BÊN DƯỚI BẢNG ---
    _, center_nav, _ = st.columns([2.6, 3.2, 2.6])
    with center_nav:
        b1, b2, b_mid, b3, b4 = st.columns([0.9, 0.9, 1.5, 0.9, 0.9], gap="small")
        with b1:
            if st.button("⏮️ Đầu", disabled=(cur_page <= 1), use_container_width=True, key="btn_first_page"):
                st.session_state["current_page"] = 1
                st.session_state["last_viewed_cell"] = None
                st.rerun()
        with b2:
            if st.button("◀️ Trước", disabled=(cur_page <= 1), use_container_width=True, key="btn_prev_page"):
                st.session_state["current_page"] -= 1
                st.session_state["last_viewed_cell"] = None
                st.rerun()
        with b_mid:
            st.markdown(
                f"<div style='text-align:center; padding-top:6px; font-size:14px; font-weight:600;'>"
                f"Trang {cur_page} / {total_pages}"
                f"</div>",
                unsafe_allow_html=True,
            )
        with b3:
            if st.button("Sau ▶️", disabled=(cur_page >= total_pages), use_container_width=True, key="btn_next_page"):
                st.session_state["current_page"] += 1
                st.session_state["last_viewed_cell"] = None
                st.rerun()
        with b4:
            if st.button("Cuối ⏭️", disabled=(cur_page >= total_pages), use_container_width=True, key="btn_last_page"):
                st.session_state["current_page"] = total_pages
                st.session_state["last_viewed_cell"] = None
                st.rerun()

    # Mở cửa sổ phóng to đúng phiếu trên trang hiện tại
    clicked_cells = event.selection.cells if event and event.selection else []
    if clicked_cells:
        cell = clicked_cells[0]
        if isinstance(cell, dict):
            row_idx, col_name = cell.get("row"), cell.get("column")
        else:
            row_idx, col_name = cell[0], cell[1]

        if col_name in ("Tiêu đề", "Nội dung") and row_idx is not None and 0 <= row_idx < len(page_tickets):
            target_ticket = page_tickets[row_idx]
            cell_signature = (target_ticket["id"], col_name)
            if st.session_state["last_viewed_cell"] != cell_signature:
                st.session_state["last_viewed_cell"] = cell_signature
                show_ticket_popup(target_ticket, enable_wait_boost=enable_wait_boost)
    else:
        st.session_state["last_viewed_cell"] = None

    # Lấy danh sách Mã phiếu đã tick trên trang hiện tại
    selected_rows = event.selection.rows if event and event.selection else []
    selected_ids = df_page.iloc[selected_rows]["Mã phiếu"].tolist() if selected_rows else []

    st.write("")
    c_del, c_csv, c_clear, _ = st.columns([2.6, 2.2, 2.0, 3.2])
    with c_del:
        if selected_ids:
            id_preview = ", ".join(f"#{i}" for i in selected_ids[:4])
            if len(selected_ids) > 4:
                id_preview += "..."
            del_label = f"❌ Xóa {len(selected_ids)} phiếu ({id_preview})"
        else:
            del_label = "❌ Xóa phiếu đã tick"

        if st.button(del_label, disabled=(len(selected_ids) == 0), use_container_width=True):
            delete_tickets(selected_ids)
            st.session_state["table_version"] += 1
            st.session_state["last_viewed_cell"] = None
            st.rerun()

    with c_csv:
        df_all_filtered = build_dataframe(filtered_tickets)
        csv_export = df_all_filtered.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            f"📥 Xuất CSV ({total_filtered})",
            data=csv_export,
            file_name="danh_sach_uu_tien_helpdesk.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with c_clear:
        if st.button("🗑️ Xóa toàn bộ", use_container_width=True):
            clear_all_tickets()
            st.session_state["table_version"] += 1
            st.session_state["last_viewed_cell"] = None
            st.session_state["current_page"] = 1
            st.rerun()


refresh_interval = 5 if (auto_refresh and enable_wait_boost) else None
st.fragment(run_every=refresh_interval)(render_priority_board)()