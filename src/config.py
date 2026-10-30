from pathlib import Path

import yaml
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent


class PipelineError(Exception):
    """Понятная пользователю ошибка (без трейсбека в консоли)."""


DEFAULTS = {
    "video_path": "", "output_dir": "output", "work_dir": "work",
    "chat_path": "", "chat_offset_sec": 0,
    "clips_count": 8, "candidates_count": 30, "auto_top_n": False,
    "clip_min_sec": 20, "clip_max_sec": 60, "language": "ru",
    "whisper_model": "large-v3", "whisper_compute_type": "float16",
    "whisper_fallback_model": "small",
    "vertical_mode": "blur", "split_top_percent": 40,
    "webcam_region": None, "game_region": None, "webcam_fit": "pad", "game_fit": "pad",
    "region_inset_px": 4, "preview_time_sec": 60,
    "subtitle_position": "boundary", "subtitle_margin_bottom": 420,
    "subtitle_font": "Arial", "subtitle_font_size": 72,
    "preroll_sec": 12, "postroll_sec": 6, "min_speech_coverage": 0.6,
    "loudness_weight": 1.0, "speech_keyword_weight": 0.7,
    "chat_spike_weight": 1.0, "chat_reaction_weight": 0.7, "speech_coverage_weight": 0.0,
    "ratings_path": "ratings.csv", "suggested_weights_path": "suggested_weights.yaml",
    "learn_min_rated": 30,
    "keywords": [], "chat_reaction_words": [],
    "llm": {"enabled": False},
}


def _resolve(p):
    path = Path(str(p)).expanduser()
    return path if path.is_absolute() else BASE_DIR / path


def _number(cfg, key, kind):
    try:
        cfg[key] = kind(cfg[key])
    except (TypeError, ValueError):
        raise PipelineError(f"В config.yaml параметр {key} должен быть числом.")


def _coord(v):
    """Число пикселей (int) или процент кадра в виде строки '27.7%'."""
    if isinstance(v, str) and v.strip().endswith("%"):
        p = float(v.strip()[:-1])
        if not 0 <= p <= 100:
            raise ValueError("процент вне диапазона 0-100")
        return f"{p:g}%"
    n = int(v)
    if n < 0:
        raise ValueError("отрицательное значение")
    return n


def _region(cfg, key, required):
    r = cfg[key]
    if r is None:
        if required:
            raise PipelineError(f"Для vertical_mode: split задай {key} в config.yaml (x, y, w, h).")
        return
    try:
        cfg[key] = {k: _coord(r[k]) for k in ("x", "y", "w", "h")}
    except (KeyError, TypeError, ValueError):
        raise PipelineError(
            f'{key} в config.yaml должен выглядеть так: {{x: "0%", y: "45%", w: "27%", h: "27%"}} '
            f"(проценты кадра, числа от 0 до 100) или {{x: 0, y: 0, w: 480, h: 270}} (пиксели).")


def load_config(path=None, need_video=True):
    load_dotenv(BASE_DIR / ".env")
    cfg_path = Path(path) if path else BASE_DIR / "config.yaml"
    if not cfg_path.exists():
        raise PipelineError(f"Не найден файл настроек: {cfg_path}")
    try:
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise PipelineError(
            f"Ошибка в config.yaml (проверь отступы и кавычки; в путях Windows "
            f"используй / вместо \\): {e}")
    if not isinstance(raw, dict):
        raise PipelineError("config.yaml должен содержать пары «ключ: значение».")

    cfg = {**DEFAULTS, **raw}
    cfg["llm"] = {**DEFAULTS["llm"], **(raw.get("llm") or {})}

    for key in ("clips_count", "candidates_count", "clip_min_sec", "clip_max_sec",
                "split_top_percent", "subtitle_margin_bottom", "subtitle_font_size", "region_inset_px",
                "learn_min_rated"):
        _number(cfg, key, int)
    for key in ("chat_offset_sec", "preview_time_sec", "preroll_sec", "postroll_sec",
                "min_speech_coverage", "loudness_weight", "speech_keyword_weight",
                "chat_spike_weight", "chat_reaction_weight", "speech_coverage_weight"):
        _number(cfg, key, float)

    if not isinstance(cfg["auto_top_n"], bool):
        raise PipelineError("auto_top_n должен быть true или false.")
    if not 1 <= cfg["clips_count"] <= 30:
        raise PipelineError("clips_count должен быть от 1 до 30.")
    if not cfg["clips_count"] <= cfg["candidates_count"] <= 100:
        raise PipelineError("Нужно: clips_count <= candidates_count <= 100.")
    if cfg["clip_min_sec"] < 5 or cfg["clip_min_sec"] >= cfg["clip_max_sec"]:
        raise PipelineError("Нужно: 5 <= clip_min_sec < clip_max_sec.")
    if cfg["vertical_mode"] not in ("blur", "crop", "split"):
        raise PipelineError("vertical_mode должен быть split, blur или crop.")
    if not 20 <= cfg["split_top_percent"] <= 60:
        raise PipelineError("split_top_percent должен быть от 20 до 60.")
    if cfg["webcam_fit"] not in ("pad", "crop"):
        raise PipelineError("webcam_fit должен быть pad или crop.")
    if cfg["game_fit"] not in ("pad", "crop"):
        raise PipelineError("game_fit должен быть pad или crop.")
    if not 0 <= cfg["region_inset_px"] <= 50:
        raise PipelineError("region_inset_px должен быть от 0 до 50.")
    if cfg["preview_time_sec"] < 0:
        raise PipelineError("preview_time_sec не может быть отрицательным.")
    if cfg["subtitle_position"] not in ("boundary", "bottom"):
        raise PipelineError("subtitle_position должен быть boundary или bottom.")
    if not 0 <= cfg["subtitle_margin_bottom"] < 1000:
        raise PipelineError("subtitle_margin_bottom должен быть от 0 до 999.")
    if cfg["preroll_sec"] < 0 or cfg["postroll_sec"] < 0:
        raise PipelineError("preroll_sec и postroll_sec не могут быть отрицательными.")
    if cfg["preroll_sec"] + cfg["postroll_sec"] > cfg["clip_max_sec"] - 10:
        raise PipelineError("preroll_sec + postroll_sec должны быть меньше clip_max_sec минимум на 10 секунд.")
    if not 0 <= cfg["min_speech_coverage"] <= 1:
        raise PipelineError("min_speech_coverage должен быть от 0 до 1.")
    if cfg["learn_min_rated"] < 5:
        raise PipelineError("learn_min_rated должен быть не меньше 5.")
    if min(cfg["loudness_weight"], cfg["speech_keyword_weight"], cfg["chat_spike_weight"],
           cfg["chat_reaction_weight"], cfg["speech_coverage_weight"]) < 0:
        raise PipelineError("Веса сигналов не могут быть отрицательными.")
    for key in ("keywords", "chat_reaction_words"):
        cfg[key] = [str(k).strip() for k in (cfg[key] or []) if str(k).strip()]

    _region(cfg, "webcam_region", required=cfg["vertical_mode"] == "split")
    _region(cfg, "game_region", required=False)

    if need_video and not cfg["video_path"]:
        raise PipelineError("В config.yaml не указан video_path.")
    video = _resolve(cfg["video_path"]) if cfg["video_path"] else None
    if need_video and not video.is_file():
        raise PipelineError(f"Видеофайл не найден: {video}")
    cfg["video_path"] = video
    for key in ("ratings_path", "suggested_weights_path"):
        cfg[key] = _resolve(cfg[key])
    cfg["chat_path"] = _resolve(cfg["chat_path"]) if str(cfg["chat_path"]).strip() else None
    for key in ("output_dir", "work_dir"):
        cfg[key] = _resolve(cfg[key])
        cfg[key].mkdir(parents=True, exist_ok=True)
    return cfg
