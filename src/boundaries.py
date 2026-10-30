import math

import numpy as np

SNAP_RADIUS = 4.0          # на сколько секунд можно сдвинуть границу к фразе или паузе
PAUSE = 0.5                # пауза между словами, которая считается границей
MIN_PRE, MIN_POST = 8, 3   # preroll и postroll не урезаются меньше этих значений при превышении длины
SILENT_Z = 0.3             # секунда считается тихой, если нет речи и громкость ниже этого уровня


class Speech:
    """Где в записи есть речь (по расшифровке) и где проходят границы фраз и пауз."""

    def __init__(self, segments, total):
        self.has_speech = bool(segments)
        m = np.zeros(int(total * 10) + 20, dtype=bool)   # речь с шагом 0.1 сек
        starts, ends = [], []
        for s in segments:
            m[int(s["start"] * 10): int(math.ceil(s["end"] * 10))] = True
            starts.append(s["start"])
            ends.append(s["end"])
            words = s.get("words") or []
            for a, b in zip(words, words[1:]):
                if b["start"] - a["end"] >= PAUSE:
                    ends.append(a["end"])
                    starts.append(b["start"])
        self.cum = np.concatenate(([0.0], np.cumsum(m)))
        self.sec = m[: total * 10].reshape(total, 10).mean(axis=1) >= 0.3   # речь по секундам
        self.starts = np.array(sorted(set(starts)))
        self.ends = np.array(sorted(set(ends)))

    def coverage(self, a, b):
        """Доля времени [a, b], занятая речью (0..1)."""
        ia, ib = int(a * 10), min(int(b * 10), len(self.cum) - 1)
        return float((self.cum[ib] - self.cum[ia]) / (ib - ia)) if ib > ia else 0.0


def peak_interval(act, t0, t1):
    """Пик в окне [t0, t1): участок вокруг максимума, где активность не ниже половины максимума.
    Возвращает (начало, конец, секунда максимума)."""
    seg = act[t0:t1]
    m = int(np.argmax(seg))
    if seg[m] <= 1e-9:
        return t0 + m, t0 + m + 1, t0 + m
    thr = 0.5 * seg[m]
    a = b = m
    while a > 0 and seg[a - 1] >= thr:
        a -= 1
    while b < len(seg) - 1 and seg[b + 1] >= thr:
        b += 1
    return t0 + a, t0 + b + 1, t0 + m


def _nearest(target, points, ok):
    """Ближайшая к target точка из points в пределах SNAP_RADIUS, для которой ok(p); иначе target."""
    if points.size == 0:
        return target
    i = np.searchsorted(points, target - SNAP_RADIUS)
    j = np.searchsorted(points, target + SNAP_RADIUS)
    best = [p for p in points[i:j] if ok(p)]
    return float(min(best, key=lambda p: abs(p - target))) if best else target


def _fit(s, e, lo, hi, total):
    s, e = max(0.0, s), min(float(total), e)
    if e - s < lo:
        e = min(float(total), s + lo)
        s = max(0.0, e - lo)
    if e - s > hi:
        e = s + hi
    return s, e


def _trim(s, e, lo, silent):
    """Обрезает тишину (нет речи и тихо) по краям, не опускаясь ниже lo."""
    i = int(s)
    while i < len(silent) and silent[i] and e - (i + 1) >= lo:
        i += 1
    if i > int(s):
        s = max(s, i - 0.5)
    j = int(math.ceil(e))
    while 0 < j <= len(silent) and silent[j - 1] and (j - 1) - s >= lo:
        j -= 1
    if j < int(math.ceil(e)):
        e = min(e, j + 0.5)
    return s, e


def interval(cfg, peak, speech, silent, total):
    """Границы клипа: пик, минус preroll, плюс postroll; сдвиг к фразе или паузе; обрезка тишины.
    При превышении clip_max_sec сначала урезается postroll (не меньше 3 сек), потом preroll (не меньше 8)."""
    ps, pe, pm = peak
    lo, hi = cfg["clip_min_sec"], cfg["clip_max_sec"]
    pre, post = cfg["preroll_sec"], cfg["postroll_sec"]
    min_pre, min_post = min(MIN_PRE, pre), min(MIN_POST, post)
    excess = (pe - ps) + pre + post - hi
    if excess > 0:
        cut = min(excess, post - min_post)
        post, excess = post - cut, excess - cut
    if excess > 0:
        cut = min(excess, pre - min_pre)
        pre, excess = pre - cut, excess - cut
    if excess > 0:   # сам пик слишком длинный: берём его часть вокруг самого яркого момента
        keep = (pe - ps) - excess
        ps = min(max(ps, pm - keep / 2), pe - keep)
        pe = ps + keep
    s, e = _fit(ps - pre, pe + post, lo, hi, total)
    # привязка к фразе не должна нарушать длину и минимальные preroll/postroll
    s = _nearest(s, speech.starts, lambda p: lo <= e - p <= hi and ps - p >= min_pre)
    e = _nearest(e, speech.ends, lambda p: lo <= p - s <= hi and p - pe >= min_post)
    s, e = _trim(s, e, lo, silent)
    return round(s, 2), round(e, 2)
