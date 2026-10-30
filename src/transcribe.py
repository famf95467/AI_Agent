import json
import logging
import os
import wave
from pathlib import Path

import numpy as np

from .config import PipelineError
from .ffmpeg_utils import run_ffmpeg

log = logging.getLogger("agent")


def video_fingerprint(video):
    st = video.stat()
    return f"{st.st_size}_{int(st.st_mtime)}"


def prepare_audio(cfg):
    """Вытаскивает моно-аудио 16 кГц. Файл привязан к конкретному видео."""
    wav = cfg["work_dir"] / f"audio_{video_fingerprint(cfg['video_path'])}.wav"
    if wav.exists() and wav.stat().st_size > 0:
        log.info("Аудио уже извлечено, использую готовое.")
        return wav
    log.info("Извлекаю аудио из видео...")
    tmp = cfg["work_dir"] / "audio_tmp.wav"
    run_ffmpeg(["-i", str(cfg["video_path"]), "-vn", "-ac", "1", "-ar", "16000",
                "-c:a", "pcm_s16le", str(tmp)])
    os.replace(tmp, wav)
    return wav


def _load_audio(path):
    """Читает наш wav (16 кГц, моно, 16 бит) в массив, чтобы faster-whisper не использовал av."""
    try:
        with wave.open(str(path), "rb") as w:
            data = w.readframes(w.getnframes())
    except (wave.Error, OSError) as e:
        raise PipelineError(f"Не удалось прочитать аудио: {e}")
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def _add_nvidia_dll_dirs():
    """Если cuBLAS/cuDNN поставлены через pip (пакеты nvidia-*), подключаем их DLL."""
    try:
        import nvidia
    except ImportError:
        return
    for base in nvidia.__path__:
        for bin_dir in Path(base).glob("*/bin"):
            try:
                os.add_dll_directory(str(bin_dir))
            except (OSError, AttributeError):
                pass
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


def _run(WhisperModel, model_name, device, compute, audio_data, cfg):
    log.info(f"Загружаю модель {model_name} ({device}, {compute}). "
             f"При первом запуске она скачается из интернета.")
    model = WhisperModel(model_name, device=device, compute_type=compute)
    seg_iter, info = model.transcribe(
        audio_data, language=cfg["language"], beam_size=5, word_timestamps=True,
        vad_filter=True, vad_parameters={"min_silence_duration_ms": 500})
    total = info.duration or 1
    out, mark = [], 10
    for s in seg_iter:
        words = [{"start": round(w.start, 2), "end": round(w.end, 2), "word": w.word}
                 for w in (s.words or [])]
        out.append({"start": round(s.start, 2), "end": round(s.end, 2),
                    "text": s.text.strip(), "words": words})
        pct = int(s.end / total * 100)
        if pct >= mark:
            log.info(f"Расшифровка: {pct}%")
            mark = pct // 10 * 10 + 10
    return out


def get_transcript(cfg, audio):
    cache = cfg["work_dir"] / (
        f"transcript_{video_fingerprint(cfg['video_path'])}_{cfg['language']}_{cfg['whisper_model']}.json")
    if cache.exists():
        try:
            log.info("Расшифровка найдена в кэше.")
            return json.loads(cache.read_text(encoding="utf-8"))
        except ValueError:
            log.warning("Кэш расшифровки повреждён, делаю заново.")

    _add_nvidia_dll_dirs()
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise PipelineError("Не установлен faster-whisper. Выполни: pip install -r requirements.txt")

    audio_data = _load_audio(audio)

    try:
        segments = _run(WhisperModel, cfg["whisper_model"], "cuda",
                        cfg["whisper_compute_type"], audio_data, cfg)
    except Exception as e:
        fb = cfg["whisper_fallback_model"]
        log.warning(f"Не удалось запустить на GPU ({e}). Перехожу на CPU, модель {fb}: будет медленнее.")
        log.debug("GPU-ошибка:", exc_info=True)
        try:
            segments = _run(WhisperModel, fb, "cpu", "int8", audio_data, cfg)
        except Exception as e2:
            log.debug("CPU-ошибка:", exc_info=True)
            raise PipelineError(f"Не удалось расшифровать речь: {e2}")

    if not segments:
        log.warning("Речь не найдена: клипы будут выбраны только по громкости и без субтитров.")
    cache.write_text(json.dumps(segments, ensure_ascii=False), encoding="utf-8")
    return segments
