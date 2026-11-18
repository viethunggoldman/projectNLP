import os
import json
import time
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from src.preprocess import prepare_phobert_text


class MultiTaskPhoBERT(nn.Module):
    def __init__(self, model_name: str, num_types: int, num_queues: int, num_prios: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.3)

        self.proj_type = nn.Sequential(
            nn.Linear(hidden, 256), nn.GELU(), nn.Dropout(0.2)
        )
        self.proj_queue = nn.Sequential(
            nn.Linear(hidden, 256), nn.GELU(), nn.Dropout(0.2)
        )
        self.proj_prio = nn.Sequential(
            nn.Linear(hidden, 256), nn.GELU(), nn.Dropout(0.2)
        )

        self.head_type = nn.Linear(256, num_types)
        self.head_queue = nn.Linear(256, num_queues)
        self.head_prio = nn.Linear(256, num_prios)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = out.last_hidden_state
        cls_emb = last_hidden[:, 0, :]
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        mean_emb = torch.sum(last_hidden * mask_expanded, 1) / torch.clamp(mask_expanded.sum(1), min=1e-9)
        pooled = self.dropout(0.5 * cls_emb + 0.5 * mean_emb)

        return (
            self.head_type(self.proj_type(pooled)),
            self.head_queue(self.proj_queue(pooled)),
            self.head_prio(self.proj_prio(pooled)),
        )


class HelpdeskPredictor:
    def __init__(self, model_dir: str = "saved_model"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        encoder_path = os.path.join(model_dir, "label_encoders.json")
        with open(encoder_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.priority_labels = meta["priority_labels"]  # ["very_low", "low", "medium", "high", "critical"]
        self.type_labels = meta["type_labels"]
        self.queue_labels = meta["queue_labels"]
        self.max_len = meta.get("max_len", 256)
        model_name = meta.get("model_name", "vinai/phobert-large")

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = MultiTaskPhoBERT(
            model_name=model_name,
            num_types=len(self.type_labels),
            num_queues=len(self.queue_labels),
            num_prios=len(self.priority_labels),
        )

        weights_path = os.path.join(model_dir, "model_weights.pt")
        state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, subject: str, body: str) -> dict:
        start_t = time.perf_counter()

        # Bước 1: Tiền xử lý chuẩn hóa teencode & tách từ pyvi
        clean_text = prepare_phobert_text(subject, body)

        # Bước 2: Mã hóa token
        enc = self.tokenizer(
            clean_text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        # Bước 3: Suy luận đồng thời 3 đầu ra
        logits_t, logits_q, logits_p = self.model(input_ids, attention_mask)

        probs_t = torch.softmax(logits_t, dim=1).squeeze(0).cpu().numpy()
        probs_q = torch.softmax(logits_q, dim=1).squeeze(0).cpu().numpy()
        probs_p = torch.softmax(logits_p, dim=1).squeeze(0).cpu().numpy()

        idx_t = int(probs_t.argmax())
        idx_q = int(probs_q.argmax())
        idx_p = int(probs_p.argmax())

        # Bước 4: Tính Urgency Score liên tục từ 1.00 đến 5.00
        weights = [1.0, 2.0, 3.0, 4.0, 5.0]
        urgency_score = sum(w * float(p) for w, p in zip(weights, probs_p))

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        return {
            "clean_text": clean_text,
            "pred_type": self.type_labels[idx_t],
            "conf_type": round(float(probs_t[idx_t]) * 100, 2),
            "pred_queue": self.queue_labels[idx_q],
            "conf_queue": round(float(probs_q[idx_q]) * 100, 2),
            "pred_priority": self.priority_labels[idx_p],
            "conf_priority": round(float(probs_p[idx_p]) * 100, 2),
            "urgency_score": round(urgency_score, 4),
            "inference_ms": round(elapsed_ms, 1),
        }