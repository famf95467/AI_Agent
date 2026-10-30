import json
import logging
import math

import numpy as np

from .config import PipelineError

log = logging.getLogger("agent")


def _norm(text):
    return text.lower().replace("ё", "е").replace("-", "")


def load_chat(cfg, total):
    """Читает JSON чата из TwitchDownloader. Возвращает (сообщений_в_секунду, реакций_в_секунду,
    (начало, конец) покрытия чата во времени видео) или None, если чата нет."""
    path = cfg["chat_path"]
    if not path:
        log.warning("Чат не задан (chat_path пуст): отбор идёт только по громкости и речи.")
        return None
    if not path.is_file():
        log.warning(f"Файл чата не найден: {path}. Отбор идёт только по громкости и речи.")
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        comments = data["comments"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise PipelineError(f"Не удалось прочитать чат {path.name} (нужен JSON из TwitchDownloader): {e}")

    words = [_norm(w) for w in cfg["chat_reaction_words"]]
    msgs, react = np.zeros(total), np.zeros(total)
    inside, offsets = 0, []
    for c in comments:
        try:
            t = math.floor(float(c["content_offset_seconds"]) + cfg["chat_offset_sec"])
            body = _norm(c["message"]["body"])
        except (KeyError, TypeError, ValueError):
            continue
        offsets.append(t)
        if 0 <= t < total:
            inside += 1
            msgs[t] += 1
            react[t] += sum(1 for w in words if w in body)
    if inside == 0:
        log.warning("Сообщения чата не попали в диапазон видео. Проверь chat_offset_sec "
                    "в config.yaml. Отбор идёт только по громкости и речи.")
        return None
    # покрытие: диапазон записи чата из файла (с учётом chat_offset_sec), иначе по последнему сообщению
    try:
        a, b = float(data["video"]["start"]), float(data["video"]["end"])
        a, b = a + cfg["chat_offset_sec"], b + cfg["chat_offset_sec"]
    except (KeyError, TypeError, ValueError):
        a = b = 0.0
    if b <= a:
        a, b = float(min(offsets)), float(max(offsets) + 1)
    a, b = max(0.0, a), min(float(total), b)
    if b <= a:
        log.warning("Чат не покрывает это видео. Проверь chat_offset_sec. Отбор идёт без чата.")
        return None
    log.info(f"Чат: {inside} из {len(comments)} сообщений попали в диапазон видео. "
             f"Покрытие чата: {int(a)}-{int(b)} сек видео.")
    return msgs, react, (a, b)
