import csv
import io
import logging

from .config import PipelineError

log = logging.getLogger("agent")
SIGNALS = ("loudness", "speech_keywords", "chat_spike", "chat_reactions", "speech_coverage")
COLUMNS = ["video", "rank", "start", "end", *SIGNALS, "rating"]
MIN_OVERLAP = 0.5   # пересечение интервалов не меньше 50% длительности (более длинного из двух)


def _fmt(v, digits=3):
    return "" if v is None else f"{v:.{digits}f}".replace(".", ",")   # запятая: чтобы Excel читал числа


def _num(text):
    text = (text or "").strip().replace(",", ".")
    try:
        return float(text) if text else None
    except ValueError:
        return None


def overlap(a0, a1, b0, b1):
    """Доля пересечения интервалов от длительности более длинного из них (0..1)."""
    inter = min(a1, b1) - max(a0, b0)
    return max(0.0, inter) / max(a1 - a0, b1 - b0, 1e-6)


def _read_rows(path):
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as e:
        raise PipelineError(f"Не удалось прочитать {path.name}: {e}")
    lines = [l for l in text.splitlines() if l.strip() and not l.lower().startswith("sep=")]
    if not lines:
        return []
    delim = ";" if ";" in lines[0] else ","
    rows = list(csv.DictReader(io.StringIO("\n".join(lines)), delimiter=delim))
    if rows and not {"video", "start", "end", "rating"} <= set(rows[0]):
        raise PipelineError(f"В {path.name} нет нужных колонок (video, start, end, rating). "
                            f"Не меняй заголовок таблицы.")
    return rows


def read_ratings(path):
    """Читает ratings.csv: список {video, rank, start, end, x: {сигнал: число|None}, rating: 0|1|2|None}."""
    if not path.exists():
        raise PipelineError(f"Нет файла {path.name}. Сначала сделай клипы (--pick) и поставь оценки.")
    out, bad = [], []
    for n, r in enumerate(_read_rows(path), start=2):
        raw = (r.get("rating") or "").strip()
        rating = None
        if raw:
            val = _num(raw)
            if val is None or val not in (0, 1, 2):
                bad.append(n)
            else:
                rating = int(val)
        out.append({"video": r.get("video", ""), "rank": r.get("rank", ""),
                    "start": _num(r.get("start")), "end": _num(r.get("end")),
                    "x": {k: _num(r.get(k)) for k in SIGNALS}, "rating": rating})
    if bad:
        raise PipelineError(f"В {path.name} оценка должна быть 0, 1 или 2. Неверные строки: "
                            f"{', '.join(map(str, bad[:10]))}.")
    return out


def _row(video, c):
    row = {"video": video, "rank": c["number"], "start": _fmt(c["start"], 2), "end": _fmt(c["end"], 2),
           "rating": ""}
    row.update({k: _fmt(c["signals"][k]["norm"]) for k in SIGNALS})
    return row


def update_ratings(cfg, candidates):
    """Добавляет сделанные клипы в ratings.csv. Строка, пересекающаяся с клипом на 50% и больше,
    обновляется, а поставленная в ней оценка сохраняется."""
    path, video = cfg["ratings_path"], cfg["video_path"].name
    rows = _read_rows(path) if path.exists() else []
    added = updated = 0
    for c in candidates:
        new = _row(video, c)
        best, best_ov = None, 0.0
        for r in rows:
            s, e = _num(r.get("start")), _num(r.get("end"))
            if r.get("video") != video or s is None or e is None:
                continue
            ov = overlap(c["start"], c["end"], s, e)
            if ov >= MIN_OVERLAP and ov > best_ov:
                best, best_ov = r, ov
        if best is not None:
            new["rating"] = best.get("rating", "")
            best.update(new)
            updated += 1
        else:
            rows.append(new)
            added += 1
    try:
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS, delimiter=";", extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    except OSError as e:
        raise PipelineError(f"Не удалось записать {path.name} (закрой его в Excel и запусти снова): {e}")
    log.info(f"{path.name}: добавлено {added}, обновлено {updated}. Поставь оценки в колонке rating "
             f"(0 - ни о чём, 1 - средне, 2 - хорошо).")
