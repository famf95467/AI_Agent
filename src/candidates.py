import json

from .config import PipelineError
from .transcribe import video_fingerprint

NAMES = {"loudness": "громкость", "speech_keywords": "слова в речи", "chat_spike": "всплеск чата",
         "chat_reactions": "реакции в чате", "speech_coverage": "доля речи"}


def _t(sec):
    s = int(sec)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60:02d}:{s % 60:02d}"


def save_candidates(cfg, candidates, chat_info):
    """Пишет candidates.json (для запуска клипов) и candidates.md (для чтения)."""
    out = cfg["output_dir"]
    meta = {"video": str(cfg["video_path"]), "fingerprint": video_fingerprint(cfg["video_path"]),
            "chat": chat_info}
    (out / "candidates.json").write_text(
        json.dumps({"meta": meta, "candidates": candidates}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    lines = ["# Кандидаты в клипы", "",
             f"Видео: {cfg['video_path'].name}. Чат: {'учтён' if chat_info['used'] else 'не использован'}.",
             "Номер = место по оценке (1 — лучший). Выбери номера и запусти: "
             "`python main.py --pick 3,7,12` (или `--pick all` для всех).", ""]
    cov = chat_info.get("coverage")
    if chat_info["used"] and cov and (cov[0] > 2 or cov[1] < chat_info["video_length"] - 2):
        lines += [f"**Внимание: чат покрывает только {_t(cov[0])}–{_t(cov[1])}.** "
                  f"Кандидаты вне этого интервала оценены без чата (оценка пересчитана по остальным сигналам).", ""]
    for c in candidates:
        sig = " · ".join(f"{NAMES[k]} {v['contribution']:.2f}" if v["contribution"] is not None
                         else f"{NAMES[k]} —" for k, v in c["signals"].items())
        lines += [f"## №{c['number']} · {c['start_hms']}–{c['end_hms']} · оценка {c['score']:.2f}",
                  f"Вклад сигналов: {sig}", f"Почему: {c['reason']}", f"> {c['speech']}", ""]
    (out / "candidates.md").write_text("\n".join(lines), encoding="utf-8")
    return out / "candidates.md"


def load_candidates(cfg):
    path = cfg["output_dir"] / "candidates.json"
    if not path.exists():
        raise PipelineError("Нет candidates.json. Сначала запусти программу без --pick.")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["meta"]["fingerprint"] != video_fingerprint(cfg["video_path"]):
            raise PipelineError("candidates.json сделан для другого видео. Запусти программу без --pick.")
        return data["candidates"]
    except (ValueError, KeyError, TypeError):
        raise PipelineError("candidates.json повреждён. Запусти программу без --pick.")


def save_clips(clips, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clips, f, ensure_ascii=False, indent=2)
