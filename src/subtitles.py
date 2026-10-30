from .config import PipelineError
from .render import split_heights

MAX_WORDS, MAX_CHARS, MAX_GAP = 4, 24, 0.7


def _t(sec):
    cs = int(round(max(sec, 0) * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _words(segments, start, end):
    out = []
    for seg in segments:
        if seg["end"] < start or seg["start"] > end:
            continue
        words = seg.get("words") or [{"start": seg["start"], "end": seg["end"], "word": seg["text"]}]
        for w in words:
            if start - 0.05 <= w["start"] < end and w["word"].strip():
                out.append({"start": w["start"], "end": min(w["end"], end), "word": w["word"].strip()})
    return out


def _group(words):
    chunks, cur = [], []
    for w in words:
        if cur:
            chars = sum(len(x["word"]) + 1 for x in cur) + len(w["word"])
            gap = w["start"] - cur[-1]["end"]
            if len(cur) >= MAX_WORDS or chars > MAX_CHARS or gap > MAX_GAP:
                chunks.append(cur)
                cur = []
        cur.append(w)
    if cur:
        chunks.append(cur)
    return chunks


def build_ass(cfg, clip, segments):
    """Пишет .ass для клипа в work_dir, возвращает имя файла (без пути)."""
    start, dur = clip["start"], clip["end"] - clip["start"]
    chunks = _group(_words(segments, clip["start"], clip["end"]))
    style = (f"Style: Default,{cfg['subtitle_font']},{cfg['subtitle_font_size']},"
             f"&H00FFFFFF,&H000000FF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,6,2,2,"
             f"60,60,{cfg['subtitle_margin_bottom']},1")
    lines = [
        "[Script Info]", "ScriptType: v4.00+", "PlayResX: 1080", "PlayResY: 1920",
        "WrapStyle: 0", "", "[V4+ Styles]",
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,"
        "Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,"
        "Alignment,MarginL,MarginR,MarginV,Encoding",
        style, "", "[Events]",
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text",
    ]
    prefix = ""
    if cfg["vertical_mode"] == "split" and cfg["subtitle_position"] == "boundary":
        # по центру границы между вебкамерой и игрой
        prefix = "{\\an5\\pos(540,%d)}" % split_heights(cfg)[0]
    for i, ch in enumerate(chunks):
        s = ch[0]["start"] - start
        e = max(ch[-1]["end"] - start, s + 0.4)
        if i + 1 < len(chunks):
            e = min(e, chunks[i + 1][0]["start"] - start)
        e = min(e, dur)
        if e <= s:
            continue
        text = " ".join(w["word"] for w in ch)
        text = text.replace("{", "(").replace("}", ")").replace("\n", " ")
        lines.append(f"Dialogue: 0,{_t(s)},{_t(e)},Default,,0,0,0,,{prefix}{text}")

    name = f"sub_{clip['index']:02d}.ass"
    try:
        (cfg["work_dir"] / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError as e:
        raise PipelineError(f"Не удалось записать субтитры: {e}")
    return name
