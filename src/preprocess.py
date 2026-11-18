import re
import pandas as pd

try:
    from pyvi import ViTokenizer
    HAS_PYVI = True
except ImportError:
    HAS_PYVI = False

TEENCODE_DICT = {
    "ko": "không",
    "k": "không",
    "dc": "được",
    "đc": "được",
    "vs": "với",
    "r": "rồi",
    "e": "em",
    "m": "mình",
    "v": "vậy",
    "ntn": "như thế nào",
    "bn": "bao nhiêu",
    "tl": "trả lời",
    "rep": "trả lời",
    "ib": "nhắn tin",
    "inb": "nhắn tin",
   #------
    "ad": "quản trị viên",
    "nv": "nhân viên",
    "kh": "khách hàng",
    "cskh": "chăm sóc khách hàng",
    "bsĩ": "bác sĩ",
    "tk": "tài khoản",
    "mk": "mật khẩu",
    "ck": "chuyển khoản",
    "sp": "sản phẩm",
    "sdt": "số điện thoại",
    "sđt": "số điện thoại",
    "đt": "điện thoại",
    "ds": "danh sách",
    "ktra": "kiểm tra",
    "t7": "thứ bảy",
    "cn": "chủ nhật",
    "vn": "việt nam",
    # --------
    "check": "kiểm tra",
    "acc": "tài khoản",
    "pass": "mật khẩu",
    "login": "đăng nhập",
    "noti": "thông báo",
    "db": "cơ sở dữ liệu",
    "prod": "production",
    "fix": "sửa lỗi",
    "up": "cập nhật",
    "dev": "lập trình viên",
    "bill": "hóa đơn",
    "fail": "thất bại",
    "full": "đầy",
    "lag": "chậm",
    "link": "liên kết",
    "add": "thêm",
    "click": "bấm",
    "search": "tìm kiếm",
    "note": "ghi chú",
}

def normalize_teencode(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<name>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()

    # Chuẩn hóa đơn vị thời gian & tiền tệ viết tắt (vd: 20p -> 20 phút, 85tr -> 85 triệu, 2tr5 -> 2,5 triệu)
    text = re.sub(r"\b(\d+)\s*p\b", r"\1 phút", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(\d+)\s*h\b", r"\1 giờ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(\d+)tr(\d+)\b", r"\1,\2 triệu", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(\d+)\s*tr\b", r"\1 triệu", text, flags=re.IGNORECASE)

    words = text.split(" ")
    normalized_words = []
    for w in words:
        # Bỏ qua mã thiết bị/tên riêng viết hoa hoặc có gạch nối (vd: TL-R600VPN, H&R, E-utilities)
        if "-" in w or "&" in w or (w.isupper() and len(w) <= 4):
            normalized_words.append(w)
            continue

        match = re.match(r"^([a-zA-ZÀ-ỹ0-9]+)([^\w\s]*)$", w)
        if match:
            core, punct = match.group(1), match.group(2)
            # Chỉ thay từ 1 ký tự (e, k, m, r, v) nếu viết thường để tránh nhầm tên viết tắt
            if len(core) == 1 and core.isupper():
                normalized_words.append(w)
                continue
            lower_core = core.lower()
            if lower_core in TEENCODE_DICT:
                core = TEENCODE_DICT[lower_core]
            normalized_words.append(core + punct)
        else:
            normalized_words.append(w)

    return " ".join(normalized_words)

def prepare_phobert_text(subject: str, body: str) -> str:
    sub_clean = normalize_teencode(str(subject) if pd.notna(subject) else "")
    bod_clean = normalize_teencode(str(body) if pd.notna(body) else "")

    # Gom gọn dấu câu lặp (vd: "!!!" -> "!", "???" -> "?")
    sub_clean = re.sub(r"([!?.,])\1+", r"\1", sub_clean)
    bod_clean = re.sub(r"([!?.,])\1+", r"\1", bod_clean)

    # Viết hoa chữ cái đầu của tiêu đề và nội dung cho PhoBERT
    if sub_clean:
        sub_clean = sub_clean[0].upper() + sub_clean[1:]
    if bod_clean:
        bod_clean = bod_clean[0].upper() + bod_clean[1:]

    if sub_clean and bod_clean:
        if sub_clean.endswith((".", "!", "?")):
            combined = f"{sub_clean} {bod_clean}"
        else:
            combined = f"{sub_clean}. {bod_clean}"
    else:
        combined = sub_clean or bod_clean

    if HAS_PYVI and combined:
        return ViTokenizer.tokenize(combined)
    return combined