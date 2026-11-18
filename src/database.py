import sqlite3
from datetime import datetime
from typing import Dict, List, Optional

DB_PATH = "tickets.db"

DEFAULT_WAIT_THRESHOLDS = {
    "critical": 20,
    "high": 40,
    "medium": 60,
    "low": 90,
    "very_low": 120,
}


def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject TEXT NOT NULL,
            body TEXT NOT NULL,
            created_at TEXT NOT NULL,
            pred_type TEXT,
            conf_type REAL,
            pred_queue TEXT,
            conf_queue REAL,
            pred_priority TEXT,
            conf_priority REAL,
            urgency_score REAL DEFAULT 0.0,
            inference_ms REAL DEFAULT 0.0
        )
    """)
    conn.commit()
    conn.close()


def insert_raw_ticket(subject: str, body: str, created_at: Optional[str] = None) -> int:
    """Bước 1: Lưu yêu cầu thô"""
    conn = get_connection()
    cursor = conn.cursor()
    ts_str = created_at if created_at else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        """
        INSERT INTO tickets (subject, body, created_at)
        VALUES (?, ?, ?)
        """,
        (subject.strip(), body.strip(), ts_str),
    )
    ticket_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return ticket_id


def update_ticket_prediction(ticket_id: int, pred: Dict):
    """Bước 2: Cập nhật kết quả phân loại, độ tin cậy và điểm AI gốc"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE tickets
        SET pred_type = ?,
            conf_type = ?,
            pred_queue = ?,
            conf_queue = ?,
            pred_priority = ?,
            conf_priority = ?,
            urgency_score = ?,
            inference_ms = ?
        WHERE id = ?
        """,
        (
            pred["pred_type"],
            pred["conf_type"],
            pred["pred_queue"],
            pred["conf_queue"],
            pred["pred_priority"],
            pred["conf_priority"],
            pred["urgency_score"],
            pred["inference_ms"],
            ticket_id,
        ),
    )
    conn.commit()
    conn.close()


def score_to_priority(score: float) -> str:
  
    if score >= 4.50:
        return "critical"
    elif score >= 3.50:
        return "high"
    elif score >= 2.50:
        return "medium"
    elif score >= 1.50:
        return "low"
    return "very_low"


def get_sorted_tickets(
    wait_thresholds: Optional[Dict[str, float]] = None,
    overdue_penalty: float = 1.5,
    enable_wait_boost: bool = True,
) -> List[Dict]:
    """
    Bước 3: Xếp hạng yêu cầu hỗ trợ:
    - Khi enable_wait_boost=False: giữ nguyên điểm AI gốc và thứ hạng ban đầu (time_bonus = 0).
    - Khi enable_wait_boost=True: cộng điểm theo tỷ lệ thời gian chờ phản hồi (tính theo giây) và tự động đôn hạng.
    """
    thresholds = {**DEFAULT_WAIT_THRESHOLDS, **(wait_thresholds or {})}

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM tickets")
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()

    now = datetime.now()
    for r in rows:
        created_dt = datetime.strptime(r["created_at"], "%Y-%m-%d %H:%M:%S")
        wait_seconds = max(0.0, (now - created_dt).total_seconds())

        base_score = float(r.get("urgency_score") or 0.0)
        base_priority = score_to_priority(base_score)
        wait_threshold = max(1.0, float(thresholds.get(base_priority, 60.0)))

        wait_ratio = wait_seconds / wait_threshold
        wait_pct = int(round(wait_ratio * 100))

        if enable_wait_boost:
            if wait_ratio <= 1.0:
                time_bonus = wait_ratio * 1.0
            else:
                time_bonus = 1.0 + (wait_ratio - 1.0) * overdue_penalty

            if wait_ratio > 1.0:
                wait_status = "🔥 Quá hạn"
            elif wait_ratio >= 0.75:
                wait_status = "⚠️ Sắp trễ"
            else:
                wait_status = "🟢 Trong hạn"
        else:
            time_bonus = 0.0
            wait_status = "⏸️ Tắt tính giờ"

        final_score = base_score + time_bonus

        r["base_priority"] = base_priority
        r["effective_priority"] = score_to_priority(final_score)
        r["wait_seconds"] = int(wait_seconds)
        r["wait_threshold"] = int(round(wait_threshold))
        r["wait_pct"] = wait_pct
        r["wait_status"] = wait_status
        r["time_bonus"] = round(time_bonus, 2)
        r["final_score"] = round(final_score, 2)

    if enable_wait_boost:
        rows.sort(key=lambda x: (-x["final_score"], -x["wait_pct"], x["created_at"]))
    else:
        rows.sort(key=lambda x: (-x["final_score"], x["created_at"]))
    return rows


def delete_tickets(ticket_ids: List[int]) -> int:
    """Xóa một hoặc nhiều bản ghi theo danh sách mã phiếu được tick"""
    if not ticket_ids:
        return 0
    conn = get_connection()
    cursor = conn.cursor()
    placeholders = ",".join("?" for _ in ticket_ids)
    cursor.execute(f"DELETE FROM tickets WHERE id IN ({placeholders})", tuple(ticket_ids))
    deleted_count = cursor.rowcount
    conn.commit()
    conn.close()
    return deleted_count


def clear_all_tickets():
    """Xóa toàn bộ dữ liệu và đặt lại bộ đếm ID về 1"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM tickets")
    cursor.execute("DELETE FROM sqlite_sequence WHERE name='tickets'")
    conn.commit()
    conn.close()