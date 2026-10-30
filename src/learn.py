import logging

import numpy as np

from .config import PipelineError
from .ratings import SIGNALS, read_ratings

log = logging.getLogger("agent")
KEYS = {"loudness": "loudness_weight", "speech_keywords": "speech_keyword_weight",
        "chat_spike": "chat_spike_weight", "chat_reactions": "chat_reaction_weight",
        "speech_coverage": "speech_coverage_weight"}
MIN_CV_CORR = 0.25   # ниже этой корреляции при проверке веса не предлагаются
MIN_MINORITY = 5     # сколько оценок должно отличаться от самой частой


def _ranks(a):
    order = np.argsort(a, kind="stable")
    r = np.empty(len(a))
    r[order] = np.arange(len(a))
    for v in np.unique(a):   # одинаковым значениям средний ранг
        m = a == v
        r[m] = r[m].mean()
    return r


def _spearman(a, b):
    ra, rb = _ranks(np.asarray(a, float)), _ranks(np.asarray(b, float))
    if ra.std() < 1e-9 or rb.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def _ridge(X, y, lam=1.0):
    Xc = X - X.mean(axis=0)
    return np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1]), Xc.T @ (y - y.mean()))


def _cv_corr(X, y, folds=5):
    """Корреляция предсказаний с оценками на отложенных частях данных (перекрёстная проверка)."""
    idx = np.random.default_rng(0).permutation(len(y))
    pred = np.zeros(len(y))
    for part in np.array_split(idx, folds):
        train = np.setdiff1d(idx, part)
        pred[part] = (X[part] - X[train].mean(axis=0)) @ _ridge(X[train], y[train])
    return _spearman(pred, y)


def run_learn(cfg, log):
    rows = read_ratings(cfg["ratings_path"])
    rated = [r for r in rows if r["rating"] is not None]
    usable = [r for r in rated if all(r["x"][k] is not None for k in SIGNALS)]
    log.info(f"Оценено клипов: {len(rated)}; без данных чата исключено: {len(rated) - len(usable)}; "
             f"в обучении: {len(usable)}.")
    need = cfg["learn_min_rated"] - len(usable)
    if need > 0:
        log.info(f"Для подбора весов нужно минимум {cfg['learn_min_rated']} оценённых клипов с данными чата. "
                 f"Ещё нужно: {need}. Ничего не изменено.")
        return

    X = np.array([[r["x"][k] for k in SIGNALS] for r in usable])
    y = np.array([r["rating"] for r in usable], dtype=float)
    values, counts = np.unique(y, return_counts=True)
    if len(values) < 2:
        log.info(f"Все оценки одинаковые (везде {int(values[0])}). По ним нельзя понять, какие сигналы "
                 f"важны: поставь разные оценки. Веса не предлагаются.")
        return
    if len(y) - counts.max() < MIN_MINORITY:
        log.info(f"Почти все оценки одинаковые: {int(values[counts.argmax()])} стоит у {counts.max()} из "
                 f"{len(y)} клипов. Для надёжного вывода нужно хотя бы {MIN_MINORITY} клипов с другой оценкой. "
                 f"Веса не предлагаются.")
        return

    old = {k: cfg[KEYS[k]] for k in SIGNALS}
    active = [i for i in range(len(SIGNALS)) if X[:, i].std() > 1e-6]
    if not active:
        log.info("Значения сигналов у всех клипов одинаковые: данных для подбора нет. Веса не предлагаются.")
        return
    Xa = X[:, active]
    corr_old = _spearman(X @ np.array([old[k] for k in SIGNALS]), y)
    corr_cv = _cv_corr(Xa, y)
    log.info(f"Корреляция оценок с текущей формулой: {corr_old:.2f}; с подобранной (при проверке на "
             f"отложенных данных): {corr_cv:.2f}.")
    if corr_cv < MIN_CV_CORR:
        log.info(f"Связь между сигналами и оценками слабая (меньше {MIN_CV_CORR}): надёжно подобрать веса "
                 f"нельзя. Добавь оценок или проверь, что они ставились по одному правилу. Веса не предлагаются.")
        return
    coef = np.clip(_ridge(Xa, y), 0, None)
    if coef.max() <= 0:
        log.info("Ни один сигнал не повышает оценку: веса не предлагаются.")
        return

    new = dict(old)
    for i, c in zip(active, coef / coef.max()):
        new[SIGNALS[i]] = round(float(c), 2)
    log.info("Веса сигналов (было -> стало):")
    for i, k in enumerate(SIGNALS):
        note = "" if i in active else "   (нет разброса в данных, не менялся)"
        if i in active and new[k] == 0:
            note = "   (связи с оценкой нет, вес 0)"
        log.info(f"  {KEYS[k]:<24}{old[k]:>5.2f} -> {new[k]:.2f}{note}")
    lines = [f"# Предложенные веса: {len(y)} оценок, корреляция при проверке {corr_cv:.2f}.",
             "# config.yaml НЕ изменён. Если согласен, перенеси значения в config.yaml вручную."]
    lines += [f"{KEYS[k]}: {new[k]:.2f}" for k in SIGNALS]
    try:
        cfg["suggested_weights_path"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError as e:
        raise PipelineError(f"Не удалось записать {cfg['suggested_weights_path'].name}: {e}")
    log.info(f"Записано: {cfg['suggested_weights_path']}. config.yaml не менялся: перенеси веса сам, если согласен.")
