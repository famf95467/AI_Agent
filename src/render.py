import logging
import math

from .config import PipelineError
from .ffmpeg_utils import probe_video, run_ffmpeg

log = logging.getLogger("agent")
W, H = 1080, 1920
MAX_SCALE = 3.0   # во сколько раз можно увеличивать область без предупреждения


def split_heights(cfg):
    """Высота верхней (вебкамера) и нижней (игра) частей, чётные числа."""
    top = int(round(H * cfg["split_top_percent"] / 100 / 2) * 2)
    return top, H - top


def _even(x):
    return max(2, int(round(x / 2)) * 2)


def _px(v, total):
    return round(float(v[:-1]) / 100 * total) if isinstance(v, str) else int(v)


def _cover_scale(r, tw, th):
    return max(tw / r["w"], th / r["h"])


def _webcam_scale(cfg, top_h):
    r = cfg["layout"]["webcam"]
    return _cover_scale(r, W, top_h) if cfg["webcam_fit"] == "crop" else W / r["w"]


def _game_scale(cfg, bottom_h):
    r = cfg["layout"]["game"]
    return _cover_scale(r, W, bottom_h) if cfg["game_fit"] == "crop" else min(W / r["w"], bottom_h / r["h"])


def prepare_layout(cfg):
    """Переводит области в пиксели, режет region_inset_px внутрь, проверяет по размеру видео."""
    vw, vh, duration = probe_video(cfg["video_path"])
    inset = cfg["region_inset_px"]
    layout = {"video": [vw, vh], "duration": duration}
    for name in ("webcam", "game"):
        src = cfg[f"{name}_region"] or {"x": 0, "y": 0, "w": "100%", "h": "100%"}
        x, y = _px(src["x"], vw), _px(src["y"], vh)
        w, h = _px(src["w"], vw), _px(src["h"], vh)
        if 0 < x + w - vw <= 2:    # допуск на округление процентов
            w = vw - x
        if 0 < y + h - vh <= 2:
            h = vh - y
        if x + w > vw or y + h > vh:
            raise PipelineError(
                f"{name}_region (x={x}, y={y}, w={w}, h={h}) выходит за пределы кадра {vw}x{vh}. "
                f"Поправь область в config.yaml.")
        inner = {"x": x + inset, "y": y + inset, "w": w - 2 * inset, "h": h - 2 * inset}
        if inner["w"] < 16 or inner["h"] < 16:
            raise PipelineError(f"{name}_region слишком мала (после отступа {inset} px внутрь).")
        layout[f"{name}_raw"], layout[name] = {"x": x, "y": y, "w": w, "h": h}, inner
    cfg["layout"] = layout

    top_h, bottom_h = split_heights(cfg)
    wc, gm = layout["webcam"], layout["game"]
    log.info(f"Раскладка split: кадр {vw}x{vh}, верх {top_h}px, низ {bottom_h}px. "
             f"Вебкамера: x={wc['x']} y={wc['y']} {wc['w']}x{wc['h']}. Игра: x={gm['x']} y={gm['y']} {gm['w']}x{gm['h']} "
             f"(с отступом {inset} px внутрь).")
    for title, scale in (("Вебкамера", _webcam_scale(cfg, top_h)), ("Игра", _game_scale(cfg, bottom_h))):
        if scale > MAX_SCALE:
            log.warning(f"{title} увеличивается в {scale:.1f} раза (больше {MAX_SCALE:.0f}x): "
                        f"качество будет низким. Выбери область крупнее или уменьши split_top_percent.")


def _blur(src, tw, th, out):
    """Размытый фон размера tw x th из потока src (считается на уменьшенной копии)."""
    return (f"[{src}]scale={tw // 4}:{th // 4}:force_original_aspect_ratio=increase,"
            f"crop={tw // 4}:{th // 4},boxblur=8:2,scale={tw}:{th}[{out}]")


def _webcam_graph(cfg, top_h):
    r = cfg["layout"]["webcam"]
    if cfg["webcam_fit"] == "crop":   # заполняем всю верхнюю часть, центр области остаётся в центре
        s = _webcam_scale(cfg, top_h)
        return [f"[wc]scale={math.ceil(r['w'] * s)}:{math.ceil(r['h'] * s)},crop={W}:{top_h}[top]"]
    sh = _even(r["h"] * W / r["w"])
    if sh >= top_h:   # не помещается по высоте: лишнее срезаем сверху и снизу
        return [f"[wc]scale={W}:{sh},crop={W}:{top_h}[top]"]
    return ["[wc]split=2[wf][wb]", _blur("wb", W, top_h, "wbg"), f"[wf]scale={W}:{sh}[wfg]",
            f"[wbg][wfg]overlay=0:{(top_h - sh) // 2}[top]"]


def _game_graph(cfg, bottom_h):
    r = cfg["layout"]["game"]
    if cfg["game_fit"] == "crop":     # заполняем всю нижнюю часть, обрезка по центру области
        s = _game_scale(cfg, bottom_h)
        return [f"[gm]scale={math.ceil(r['w'] * s)}:{math.ceil(r['h'] * s)},crop={W}:{bottom_h}[bottom]"]
    s = min(W / r["w"], bottom_h / r["h"])   # целиком, без обрезки
    fw, fh = min(_even(r["w"] * s), W), min(_even(r["h"] * s), bottom_h)
    if fw == W and fh == bottom_h:
        return [f"[gm]scale={W}:{bottom_h}[bottom]"]
    return ["[gm]split=2[gf][gb]", _blur("gb", W, bottom_h, "gbg"), f"[gf]scale={fw}:{fh}[gfg]",
            f"[gbg][gfg]overlay={(W - fw) // 2}:{(bottom_h - fh) // 2}[bottom]"]


def split_parts(cfg, src="0:v"):
    """Фильтры раскладки split от потока src до метки [st] (кадр 1080x1920 без субтитров)."""
    top_h, bottom_h = split_heights(cfg)
    w, g = cfg["layout"]["webcam"], cfg["layout"]["game"]
    parts = [f"[{src}]split=2[v1][v2]",
             f"[v1]crop={w['w']}:{w['h']}:{w['x']}:{w['y']}[wc]",
             f"[v2]crop={g['w']}:{g['h']}:{g['x']}:{g['y']}[gm]"]
    parts += _webcam_graph(cfg, top_h) + _game_graph(cfg, bottom_h)
    parts.append("[top][bottom]vstack=inputs=2,setsar=1[st]")
    return parts


def _filter(cfg, ass_name):
    mode = cfg["vertical_mode"]
    if mode == "split":
        return ";".join(split_parts(cfg) + [f"[st]ass={ass_name}[v]"])
    if mode == "blur":
        return (
            "[0:v]split=2[bg][fg];"
            "[bg]scale=270:480:force_original_aspect_ratio=increase,crop=270:480,"
            "boxblur=10:2,scale=1080:1920[b];"
            "[fg]scale=1080:1920:force_original_aspect_ratio=decrease[f];"
            f"[b][f]overlay=(W-w)/2:(H-h)/2,setsar=1,ass={ass_name}[v]")
    return ("[0:v]scale=1080:1920:force_original_aspect_ratio=increase,"
            f"crop=1080:1920,setsar=1,ass={ass_name}[v]")


def _hms(sec):
    s = int(sec)
    return f"{s // 3600:02d}h{s % 3600 // 60:02d}m{s % 60:02d}s"


def clip_filename(index, start, end):
    return f"rank_{index:02d}_{_hms(start)}-{_hms(end)}.mp4"


def render_clip(cfg, clip, ass_name):
    """Вырезает клип, делает 1080x1920 и вшивает субтитры. Возвращает путь к mp4."""
    out = cfg["output_dir"] / clip_filename(clip["index"], clip["start"], clip["end"])
    dur = clip["end"] - clip["start"]
    args = [
        "-ss", f"{clip['start']:.3f}", "-i", str(cfg["video_path"]), "-t", f"{dur:.3f}",
        "-filter_complex", _filter(cfg, ass_name),
        "-map", "[v]", "-map", "0:a:0?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out),
    ]
    # cwd = work_dir: так .ass указывается коротким именем, без проблем с путями Windows
    run_ffmpeg(args, cwd=str(cfg["work_dir"]))
    if not out.exists():
        raise PipelineError(f"ffmpeg не создал файл {out.name}")
    return out
