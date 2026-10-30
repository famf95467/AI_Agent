import json
import logging
import shutil
import subprocess

from .config import PipelineError

log = logging.getLogger("agent")


def check_ffmpeg():
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise PipelineError(
                f"{tool} не найден. Установи ffmpeg (например: winget install Gyan.FFmpeg), "
                f"перезапусти консоль и проверь командой: {tool} -version")


def run_ffmpeg(args, cwd=None):
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"] + args
    log.debug("ffmpeg: " + " ".join(cmd))
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, stdin=subprocess.DEVNULL,
                           encoding="utf-8", errors="replace")
    except OSError as e:
        raise PipelineError(f"Не удалось запустить ffmpeg: {e}")
    if r.returncode != 0:
        tail = "\n".join(r.stderr.strip().splitlines()[-5:])
        raise PipelineError(f"ffmpeg завершился с ошибкой:\n{tail}")


def probe_video(path):
    """Ширина, высота и длительность (сек) видео."""
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height:format=duration", "-of", "json", str(path)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL,
                           encoding="utf-8", errors="replace")
        data = json.loads(r.stdout)
        st = data["streams"][0]
        return int(st["width"]), int(st["height"]), float(data["format"].get("duration") or 0)
    except (OSError, ValueError, KeyError, IndexError):
        raise PipelineError("Не удалось определить размер видео (ffprobe). Файл повреждён или это не видео?")
