import argparse
import sys
from pathlib import Path

from src import llm_stub
from src.candidates import load_candidates, save_candidates, save_clips
from src.config import PipelineError, load_config
from src.ffmpeg_utils import check_ffmpeg
from src.highlights import find_candidates
from src.learn import run_learn
from src.logger import setup_logger
from src.preview import confirm_layout
from src.ratings import update_ratings
from src.render import prepare_layout, render_clip
from src.subtitles import build_ass
from src.transcribe import get_transcript, prepare_audio


def parse_pick(text):
    if text.strip().lower() == "all":
        return "all"
    try:
        nums = sorted({int(x) for x in text.split(",") if x.strip()})
    except ValueError:
        nums = []
    if not nums:
        raise PipelineError("--pick: укажи номера через запятую (--pick 3,7,12) или all (все кандидаты)")
    return nums


def render_all(cfg, chosen, segments, log):
    log.info(f"Нарезка и субтитры: клипов {len(chosen)}")
    clips, made = [], []
    for c in chosen:
        clip = {"index": c["number"], "start": c["start"], "end": c["end"],
                "start_hms": c["start_hms"], "end_hms": c["end_hms"],
                "score": c["score"], "reason": c["reason"], "file": None}
        try:
            ass_name = build_ass(cfg, clip, segments)
            clip["file"] = render_clip(cfg, clip, ass_name).name
            made.append(c)
            log.info(f"Клип №{clip['index']} готов ({len(made)}/{len(chosen)}): {clip['file']}")
        except PipelineError as e:
            log.error(f"Клип №{clip['index']} пропущен: {e}")
        clips.append(clip)
    save_clips(clips, cfg["output_dir"] / "clips.json")
    if not made:
        raise PipelineError("Ни один клип не создан. Причины выше и в run.log.")
    log.info(f"Готово: {len(made)} из {len(chosen)} клипов в папке {cfg['output_dir']}")
    try:
        update_ratings(cfg, made)
    except PipelineError as e:   # клипы уже готовы, поэтому только сообщаем
        log.error(f"Клипы готовы, но таблица оценок не обновлена: {e}")


def run(log, pick, preview_only, learn):
    if learn:
        run_learn(load_config(need_video=False), log)
        return
    cfg = load_config()
    check_ffmpeg()
    split = cfg["vertical_mode"] == "split"
    if split:
        prepare_layout(cfg)
    if preview_only:
        if not split:
            raise PipelineError("Превью нужно только для vertical_mode: split.")
        confirm_layout(cfg, log, force=True)
        return
    if split and (pick is not None or cfg["auto_top_n"]):
        confirm_layout(cfg, log)   # без подтверждения превью нарезка не начнётся
    log.info(f"Видео: {cfg['video_path']}")

    log.info("Шаг 1: расшифровка речи")
    audio = prepare_audio(cfg)
    segments = get_transcript(cfg, audio)

    if pick is not None:   # быстрый путь: кандидаты уже посчитаны, делаем только клипы
        by_num = {c["number"]: c for c in load_candidates(cfg)}
        nums = sorted(by_num) if pick == "all" else pick
        missing = [n for n in nums if n not in by_num]
        if missing:
            raise PipelineError(f"Нет кандидатов с номерами {missing}. Доступны 1–{len(by_num)}.")
        render_all(cfg, [by_num[n] for n in nums], segments, log)
        return

    log.info("Шаг 2: поиск кандидатов")
    candidates, chat_info = None, {"used": False, "coverage": None, "video_length": 0}
    if cfg["llm"]["enabled"]:
        try:
            candidates = llm_stub.select_moments(cfg, segments)
        except NotImplementedError as e:
            log.warning(str(e))
    if not candidates:
        candidates, chat_info = find_candidates(cfg, segments, audio)
    path = save_candidates(cfg, candidates, chat_info)
    log.info(f"Кандидатов: {len(candidates)}. Список: {path}")

    if cfg["auto_top_n"]:
        render_all(cfg, candidates[:cfg["clips_count"]], segments, log)
    else:
        log.info("Открой candidates.md, выбери номера и запусти: python main.py --pick 3,7,12 (или --pick all)")
        if split:
            log.info("Раскладку можно проверить заранее: python main.py --preview")


def main():
    parser = argparse.ArgumentParser(description="Контент-агент: нарезка стримов")
    parser.add_argument("--pick", help="номера кандидатов через запятую (3,7,12) или all - все")
    parser.add_argument("--preview", action="store_true",
                        help="сохранить output/preview.png и подтвердить раскладку split, без нарезки")
    parser.add_argument("--learn", action="store_true",
                        help="подобрать веса сигналов по оценкам из ratings.csv (config.yaml не меняется)")
    args = parser.parse_args()
    if args.learn and (args.pick or args.preview):
        parser.error("--learn нельзя сочетать с --pick и --preview")
    log = setup_logger(Path(__file__).resolve().parent / "run.log")
    try:
        pick = parse_pick(args.pick) if args.pick is not None else None
        run(log, pick, args.preview, args.learn)
    except PipelineError as e:
        log.error(f"ОШИБКА: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        log.error("Остановлено пользователем.")
        sys.exit(1)
    except Exception as e:  # трейсбек только в файл
        log.debug("Трейсбек:", exc_info=True)
        log.error(f"Непредвиденная ошибка: {e}. Подробности в run.log")
        sys.exit(1)


if __name__ == "__main__":
    main()
