import hashlib
import json

from .config import PipelineError
from .ffmpeg_utils import run_ffmpeg
from .render import W, split_heights, split_parts


def _layout_key(cfg):
    """Отпечаток раскладки: если он изменился, превью нужно подтвердить заново."""
    data = {k: cfg[k] for k in ("split_top_percent", "webcam_fit", "game_fit", "region_inset_px")}
    data["layout"] = {k: v for k, v in cfg["layout"].items() if k != "duration"}
    return hashlib.sha1(json.dumps(data, sort_keys=True).encode()).hexdigest()


def make_preview(cfg):
    """preview.png: слева исходный кадр с рамками областей, справа итоговый вертикальный кадр."""
    L = cfg["layout"]
    top_h, _ = split_heights(cfg)
    t = min(cfg["preview_time_sec"], max(L["duration"] - 1, 0))
    boxes = ",".join(
        f"drawbox=x={r['x']}:y={r['y']}:w={r['w']}:h={r['h']}:color={color}:t=4"
        for r, color in ((L["webcam_raw"], "red"), (L["game_raw"], "lime")))
    graph = ";".join(
        ["[0:v]split=2[src][v0]", f"[src]{boxes},scale=-2:960[left]"]
        + split_parts(cfg, "v0")
        + [f"[st]drawbox=x=0:y={top_h - 1}:w={W}:h=2:color=yellow:t=fill,scale=-2:960[right]",
           "[left][right]hstack=inputs=2[p]"])
    out = cfg["output_dir"] / "preview.png"
    run_ffmpeg(["-ss", f"{t:.3f}", "-i", str(cfg["video_path"]), "-filter_complex", graph,
                "-map", "[p]", "-frames:v", "1", str(out)])
    return out


def confirm_layout(cfg, log, force=False):
    """Сохраняет preview.png и просит подтверждение. Без подтверждения нарезка не начнётся.
    Если раскладка не менялась и уже подтверждена, вопрос не задаётся (кроме force)."""
    marker = cfg["work_dir"] / "layout_confirmed.json"
    key = _layout_key(cfg)
    path = make_preview(cfg)
    try:
        confirmed = json.loads(marker.read_text(encoding="utf-8")).get("key") == key
    except (OSError, ValueError, AttributeError):
        confirmed = False
    if confirmed and not force:
        log.info(f"Превью обновлено: {path}. Раскладка не менялась и уже подтверждена.")
        return
    log.info(f"Превью сохранено: {path}")
    log.info("Слева исходный кадр: красная рамка - вебкамера, зелёная - игра. "
             "Справа итоговый вертикальный кадр (жёлтая линия - граница частей).")
    try:
        answer = input("Превью верное? Нарезка начнётся только после подтверждения (да/нет): ")
    except EOFError:
        raise PipelineError("Нужно подтвердить превью в консоли. Запусти: python main.py --preview")
    if answer.strip().lower() not in ("да", "д", "y", "yes"):
        raise PipelineError("Превью не подтверждено, нарезка не запущена. "
                            "Поправь области в config.yaml и запусти: python main.py --preview")
    marker.write_text(json.dumps({"key": key}), encoding="utf-8")
    log.info("Превью подтверждено.")
