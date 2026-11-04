"""Locate ffmpeg: a previously downloaded build, then a usable one on PATH, else download one.

Downloads are static builds for this platform: BtbN's latest stable release for Linux and Windows
(with NVENC), and Martin Riedl's release builds for macOS (Apple silicon or Intel). The binary is
kept in ~/.EAGLE/ffmpeg/, so the download happens once.
"""

import io
import json
import os
import platform
import re
import shutil
import ssl
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

from filelock import FileLock

BTBN_LATEST_RELEASE_API = "https://api.github.com/repos/BtbN/FFmpeg-Builds/releases/tags/latest"
MACOS_BUILD_URL = "https://ffmpeg.martin-riedl.de/redirect/latest/macos/{arch}/release/ffmpeg.zip"

_resolved: dict[Path, str] = {}


def ffmpeg_executable(target: Path) -> str:
    """Return a usable ffmpeg executable, downloading a build to `target` when none is available."""
    if target in _resolved:
        return _resolved[target]
    executable = None
    if target.is_file():
        executable = str(target)
    else:
        system = shutil.which("ffmpeg")
        if system is not None and _has_libx264(system):
            executable = system
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            # Several EAGLE processes may start at once; only one downloads.
            with FileLock(str(target.parent / "download.lock")):
                if not target.is_file():
                    try:
                        _download(target)
                    except Exception as exc:
                        if system is None:
                            raise RuntimeError(
                                "ffmpeg was not found and could not be downloaded.\n"
                                "Connect this machine to the internet and try again, or install ffmpeg "
                                "(for example with your package manager) so that it is on PATH.\n"
                                f"Original error: {exc}"
                            ) from exc
                        print(f"Could not download ffmpeg ({exc}); using {system}.", flush=True)
                        executable = system
            executable = executable or str(target)
    _resolved[target] = executable
    return executable


def _has_libx264(executable: str) -> bool:
    try:
        result = subprocess.run([executable, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=30)
    except Exception:
        return False
    return result.returncode == 0 and "libx264" in result.stdout


def _ssl_context() -> ssl.SSLContext | None:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return None


def _fetch(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "EAGLE"})
    with urlopen(request, context=_ssl_context(), timeout=600) as response:
        return response.read()


def _download_url() -> tuple[str, str]:
    """The archive URL for this platform and the name of the ffmpeg executable inside it."""
    machine = platform.machine().lower()
    arm = machine in {"arm64", "aarch64"}
    if sys.platform == "darwin":
        return MACOS_BUILD_URL.format(arch="arm64" if arm else "amd64"), "ffmpeg"
    if sys.platform.startswith("win"):
        build, executable = "win64", "ffmpeg.exe"
    elif sys.platform.startswith("linux"):
        build, executable = ("linuxarm64" if arm else "linux64"), "ffmpeg"
    else:
        raise RuntimeError(f"No ffmpeg build is known for {sys.platform} ({machine}).")
    release = json.loads(_fetch(BTBN_LATEST_RELEASE_API))
    assets = {asset["name"]: asset["browser_download_url"] for asset in release.get("assets", [])}
    # Prefer the newest stable release branch (e.g. n9.0) over master builds.
    stable = []
    for name in assets:
        match = re.fullmatch(rf"ffmpeg-n([\d.]+)-latest-{build}-gpl-[\d.]+\.(?:tar\.xz|zip)", name)
        if match:
            stable.append((tuple(int(part) for part in match.group(1).split(".")), name))
    if stable:
        return assets[max(stable)[1]], executable
    for name, url in assets.items():
        if re.fullmatch(rf"ffmpeg-master-latest-{build}-gpl\.(?:tar\.xz|zip)", name):
            return url, executable
    raise RuntimeError(f"No ffmpeg build for {build} was found in {BTBN_LATEST_RELEASE_API}.")


def _download(target: Path) -> None:
    url, executable = _download_url()
    print(f"Downloading ffmpeg for this platform from {url} (first run only)...", flush=True)
    archive = _fetch(url)
    if url.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
            member = next(name for name in zipped.namelist() if Path(name).name == executable)
            data = zipped.read(member)
    else:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:xz") as tarred:
            member = next(info for info in tarred.getmembers() if Path(info.name).name == executable and info.isfile())
            data = tarred.extractfile(member).read()
    partial = target.with_name(target.name + ".part")
    partial.write_bytes(data)
    partial.chmod(0o755)
    os.replace(partial, target)
    print(f"ffmpeg saved to {target}", flush=True)
