import logging
import math
import wave

import numpy as np

from .boundaries import SILENT_Z, Speech, interval, peak_interval
from .chat import load_chat
from .config import PipelineError

log = logging.getLogger("agent")
STEP = 5         # шаг скользящего окна, сек
TOP_N = 5        # сколько самых громких секунд учитываем в окне
GAP = 10         # минимальный промежуток между кандидатами, сек
CHAT_WIN = 10    # окно для подсчёта всплеска чата, сек
SIGNALS = ("loudness", "speech_keywords", "chat_spike", "chat_reactions", "speech_coverage")
CHAT_SIGNALS = ("chat_spike", "chat_reactions")


def _norm(text):
    return text.lower().replace("ё", "е").replace("-", "")


def _keyword_hits(text, keywords):
    t = _norm(text)
    found = {}
    for kw in keywords:
        n = t.count(_norm(kw))
        if n:
            found[kw] = n
    return found


def _loudness(wav_path):
    """Громкость по секундам в виде z-оценки (0 = обычный уровень стрима)."""
    levels = []
    try:
        with wave.open(str(wav_path), "rb") as w:
            sr = w.getframerate()
            if w.getsampwidth() != 2 or w.getnchannels() != 1:
                raise PipelineError("Неожиданный формат аудио. Удали папку work и запусти снова.")
            while True:
                data = w.readframes(sr * 60)
                if not data:
                    break
                x = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                n = len(x) // sr
                if n == 0:
                    continue
                x = x[: n * sr].reshape(n, sr)
                rms = np.sqrt((x * x).mean(axis=1))
                levels.append(20 * np.log10(rms + 1.0))
    except (wave.Error, OSError) as e:
        raise PipelineError(f"Не удалось прочитать аудио: {e}")
    if not levels:
        raise PipelineError("Аудиодорожка пустая.")
    db = np.concatenate(levels)
    return np.clip((db - np.median(db)) / (db.std() + 1e-6), 0, 5)


def _chat_spike_z(msgs, cov):
    """z-оценка числа сообщений в окне CHAT_WIN сек (индекс t = начало окна).
    Среднее и разброс считаются только там, где чат покрывает видео."""
    if len(msgs) < CHAT_WIN:
        return np.zeros(1)
    csum = np.concatenate(([0.0], np.cumsum(msgs)))
    s = csum[CHAT_WIN:] - csum[:-CHAT_WIN]
    lo_i, hi_i = max(0, int(math.ceil(cov[0]))), min(len(s), int(cov[1]) - CHAT_WIN + 1)
    z = np.zeros(len(s))
    if hi_i - lo_i >= 2:
        ref = s[lo_i:hi_i]
        z[lo_i:hi_i] = np.clip((ref - ref.mean()) / (ref.std() + 1e-6), 0, 5)
    return z


def _max(arr, a, b):
    part = arr[max(a, 0):max(b, 0)]
    return float(part.max()) if part.size else 0.0


def _hms(sec):
    s = int(sec)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def _excerpt(segments, a, b, peak):
    """2-3 фразы из расшифровки, ближайшие к самому громкому моменту."""
    segs = [x for x in segments if a <= x["start"] < b and len(x["text"]) >= 10]
    segs.sort(key=lambda x: abs((x["start"] + x["end"]) / 2 - peak))
    pick = sorted(segs[:3], key=lambda x: x["start"])
    return " … ".join(x["text"][:200] for x in pick) or "(речи нет)"


def _reason(r, found, chat_ok):
    parts = []
    if r["loudness"] >= 1.0:
        parts.append(f"всплеск громкости (+{r['loudness']:.1f}σ)")
    if found:
        parts.append("слова в речи: " + ", ".join(f"{k} ×{v}" for k, v in found.items()))
    if chat_ok and r["chat_spike"] >= 1.0:
        parts.append(f"всплеск чата (+{r['chat_spike']:.1f}σ)")
    if chat_ok and r["chat_reactions"] > 0:
        parts.append(f"реакций в чате: {int(r['chat_reactions'])}")
    text = "; ".join(parts) or "самый оживлённый из оставшихся фрагментов"
    return f"{text} (речь {r['speech_coverage']:.0%})"


def find_candidates(cfg, segments, audio_path):
    """Возвращает (кандидаты, сведения о чате). Номер 1 = лучший по оценке."""
    z = _loudness(audio_path)
    total = len(z)
    lo, hi = cfg["clip_min_sec"], cfg["clip_max_sec"]
    if total < lo:
        raise PipelineError(f"Видео ({total} сек) короче минимальной длины клипа ({lo} сек).")
    length = min((lo + hi) // 2, total)
    speech = Speech(segments, total)
    if not speech.has_speech:
        log.warning("Расшифровка пуста: доля речи не проверяется.")

    kw_sec = np.zeros(total)
    for sg in segments:
        n = sum(_keyword_hits(sg["text"], cfg["keywords"]).values())
        if n:
            kw_sec[min(int((sg["start"] + sg["end"]) / 2), total - 1)] += n

    chat = load_chat(cfg, total)
    if chat is None:
        cov, react, spike_z = None, np.zeros(total), np.zeros(1)
    else:
        msgs, react, cov = chat
        spike_z = _chat_spike_z(msgs, cov)

    def chat_ok(a, b):   # чат учитывается, только если клип целиком внутри его покрытия
        return cov is not None and cov[0] <= a and b <= cov[1]

    weights = {"loudness": cfg["loudness_weight"], "speech_keywords": cfg["speech_keyword_weight"],
               "chat_spike": cfg["chat_spike_weight"], "chat_reactions": cfg["chat_reaction_weight"],
               "speech_coverage": cfg["speech_coverage_weight"]}

    # активность по секундам (громкость + чат): по ней ищется пик
    has_chat = np.zeros(total, dtype=bool)
    if cov is not None:
        has_chat[int(math.ceil(cov[0])):int(cov[1])] = True
    cn = spike_z[np.clip(np.arange(total) - CHAT_WIN // 2, 0, len(spike_z) - 1)] / 5
    wl, wc = weights["loudness"], weights["chat_spike"]
    if wl + wc <= 0:
        wl = 1.0
    act = np.where(has_chat, (wl * z / 5 + wc * cn) / (wl + wc), z / 5)
    act = np.convolve(act, np.ones(3) / 3, mode="same")
    silent = ~speech.sec & (z < SILENT_Z)

    def raw(a, b):
        ia, ib = int(a), int(b)
        top = np.sort(z[ia:ib])[-TOP_N:]
        return {"loudness": float(top.mean()) if top.size else 0.0,
                "speech_keywords": float(kw_sec[ia:ib].sum()),
                "chat_spike": _max(spike_z, ia, ib - CHAT_WIN + 1),
                "chat_reactions": float(react[ia:ib].sum()),
                "speech_coverage": speech.coverage(a, b)}

    starts = list(range(0, total - length + 1, STEP))
    wins = [raw(t, t + length) for t in starts]
    maxes = {k: max(w[k] for w in wins) for k in SIGNALS}

    def evaluate(r, ok):
        """Нормированные значения и вклады доступных сигналов. Вклады делятся на сумму весов
        доступных сигналов: без чата оценка пересчитывается по остальным."""
        avail = [k for k in SIGNALS if ok or k not in CHAT_SIGNALS]
        wsum = sum(weights[k] for k in avail)
        norm = {k: r[k] if k == "speech_coverage" else (min(1.0, r[k] / maxes[k]) if maxes[k] > 0 else 0.0)
                for k in avail}
        return norm, {k: weights[k] * norm[k] / wsum if wsum > 0 else 0.0 for k in avail}

    scores = np.array([sum(evaluate(w, chat_ok(t, t + length))[1].values()) for w, t in zip(wins, starts)])
    chosen, rejected = [], set()
    for i in np.argsort(-scores):
        if len(chosen) >= cfg["candidates_count"]:
            break
        t = starts[i]
        s0, e0 = interval(cfg, peak_interval(act, t, t + length), speech, silent, total)
        if any(s0 < c[1] + GAP and e0 > c[0] - GAP for c in chosen):
            continue
        if speech.has_speech and speech.coverage(s0, e0) < cfg["min_speech_coverage"]:
            rejected.add((round(s0), round(e0)))   # мало речи: берём следующего по оценке
            continue
        chosen.append((s0, e0))
    if rejected:
        log.info(f"Отброшено из-за малой доли речи (меньше {cfg['min_speech_coverage']:.0%}): "
                 f"{len(rejected)}, заменены следующими по оценке.")
    if len(chosen) < cfg["candidates_count"]:
        log.warning(f"Найдено только {len(chosen)} подходящих кандидатов из {cfg['candidates_count']}.")

    cands = []
    for s0, e0 in chosen:
        r, ok = raw(s0, e0), chat_ok(s0, e0)
        norm, contrib = evaluate(r, ok)
        found = {}
        for sg in segments:
            if s0 <= sg["start"] < e0:
                for k, v in _keyword_hits(sg["text"], cfg["keywords"]).items():
                    found[k] = found.get(k, 0) + v
        peak = s0 + int(np.argmax(z[int(s0):int(e0)]))
        empty = {"raw": None, "norm": None, "contribution": None}
        cands.append({
            "start": s0, "end": e0, "duration": round(e0 - s0, 2),
            "start_hms": _hms(s0), "end_hms": _hms(e0),
            "score": round(sum(contrib.values()), 3), "chat_available": ok,
            "signals": {k: ({"raw": round(r[k], 3), "norm": round(norm[k], 3),
                             "contribution": round(contrib[k], 3)} if k in norm else empty) for k in SIGNALS},
            "speech": _excerpt(segments, s0, e0, peak), "reason": _reason(r, found, ok),
        })
    cands.sort(key=lambda x: -x["score"])
    info = {"used": chat is not None, "coverage": list(cov) if cov else None, "video_length": total}
    return [{"number": n, **c} for n, c in enumerate(cands, 1)], info
