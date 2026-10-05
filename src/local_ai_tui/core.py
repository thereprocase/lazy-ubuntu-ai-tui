from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
import ipaddress
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import urlparse

LOG = logging.getLogger("local_ai_tui")
DEFAULT_REPO = "unsloth/Qwen3.8-Flash-Next-GGUF"
DEFAULT_QUANT = "UD-Q4_K_XL"
GPU_QUERY = "index,uuid,name,memory.total,compute_cap"


@dataclass(frozen=True)
class GPU:
    index: int
    uuid: str
    name: str
    memory_mib: int
    compute_cap: str

    @property
    def architecture(self) -> str:
        try:
            major, minor = (int(x) for x in self.compute_cap.split(".", 1))
        except (ValueError, AttributeError):
            return "unknown"
        if major in {10, 12}:
            return "Blackwell"
        if major > 12 or major == 11:
            return "newer NVIDIA architecture"
        if major == 9:
            return "Hopper"
        if major == 8:
            return "Ada" if minor == 9 else "Ampere"
        if major == 7:
            return "Turing" if minor == 5 else "Volta"
        if major == 6:
            return "Pascal"
        if major == 5:
            return "Maxwell"
        return "legacy/unknown"


@dataclass
class Config:
    install_dir: str = str(Path.home() / "local-ai-stack")
    model_path: str = ""
    mmproj_path: str = ""
    repo_id: str = DEFAULT_REPO
    quant: str = DEFAULT_QUANT
    gpu_uuids: list[str] = field(default_factory=list)
    context: int = 8192
    webui_port: int = 3000
    split_mode: str = "layer"
    tensor_split: str = ""
    gpu_layers: int | None = None
    llama_image: str = "ghcr.io/ggml-org/llama.cpp:server-cuda"
    webui_image: str = "ghcr.io/open-webui/open-webui:main"
    tailscale_mode: str = "off"  # off, serve
    tailscale_https_port: int = 443
    tailscale_expected_url: str = ""
    remote_endpoint_url: str = ""
    remote_api_key: str = field(default="", repr=False)
    complexity: str = "simple"


def configure_logging() -> Path:
    base = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state")) / "local-ai-tui"
    base.mkdir(parents=True, exist_ok=True)
    path = base / "setup.log"
    handler = RotatingFileHandler(path, maxBytes=5_000_000, backupCount=4, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOG.setLevel(logging.DEBUG)
    LOG.addHandler(handler)
    return path


def run(args: list[str], *, check: bool = True, timeout: int | None = 60) -> subprocess.CompletedProcess[str]:
    LOG.info("Running: %s", " ".join(args))
    result = subprocess.run(args, text=True, capture_output=True, timeout=timeout, check=False)
    if result.stdout:
        LOG.debug("stdout: %s", result.stdout[-10000:])
    if result.stderr:
        LOG.debug("stderr: %s", result.stderr[-10000:])
    LOG.info("Exit code: %s", result.returncode)
    if check and result.returncode:
        raise RuntimeError(f"{' '.join(args[:3])} failed ({result.returncode}): {result.stderr[-1000:]}")
    return result


def detect_gpus() -> list[GPU]:
    if not shutil.which("nvidia-smi"):
        return []
    try:
        output = run(["nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"]).stdout
    except (RuntimeError, subprocess.TimeoutExpired) as exc:
        LOG.warning("GPU detection failed: %s", exc)
        return []
    found = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            LOG.warning("Unrecognized nvidia-smi line: %s", line)
            continue
        try:
            found.append(GPU(int(parts[0]), parts[1], parts[2], int(parts[3]), parts[4]))
        except ValueError:
            LOG.warning("Unrecognized nvidia-smi values: %s", line)
    return found


def check_prerequisites() -> list[str]:
    issues = []
    if not shutil.which("docker"):
        issues.append("Docker CLI is missing")
    else:
        for command, label in [(["docker", "compose", "version"], "Docker Compose"), (["docker", "info", "--format", "{{.ServerVersion}}"], "Docker daemon")]:
            try:
                run(command)
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                issues.append(f"{label} unavailable: {exc}")
    if not detect_gpus():
        issues.append("No NVIDIA GPUs detected by nvidia-smi")
    return issues


def validate(config: Config, gpus: list[GPU]) -> list[str]:
    errors = []
    if not Path(config.model_path).is_file() or Path(config.model_path).suffix.lower() != ".gguf":
        errors.append("Select a local .gguf file (the first shard for split files)")
    if config.mmproj_path and not Path(config.mmproj_path).is_file():
        errors.append("Vision projector file does not exist")
    if not config.gpu_uuids:
        errors.append("Select at least one GPU")
    available = {gpu.uuid for gpu in gpus}
    if set(config.gpu_uuids) - available:
        errors.append("Selected GPU UUID is no longer present")
    if not 512 <= config.context <= 1_000_000:
        errors.append("Context must be 512–1,000,000")
    if not 1 <= config.webui_port <= 65535:
        errors.append("WebUI port must be 1–65535")
    if config.split_mode not in {"layer", "row", "none"}:
        errors.append("Split mode must be layer, row or none")
    if config.gpu_layers is not None and not 0 <= config.gpu_layers <= 999:
        errors.append("GPU layers must be 0–999")
    if config.tensor_split and not re.fullmatch(r"\d+(?:\.\d+)?(?:,\d+(?:\.\d+)?)*", config.tensor_split):
        errors.append("Tensor split must be comma-separated nonnegative numbers")
    if config.tensor_split and len(config.tensor_split.split(",")) != len(config.gpu_uuids):
        errors.append("Tensor split needs one value per selected GPU")
    if config.tailscale_mode not in {"off", "serve"}:
        errors.append("Tailscale mode must be off or serve")
    if not 1 <= config.tailscale_https_port <= 65535:
        errors.append("Tailscale HTTPS port must be 1–65535")
    if config.tailscale_expected_url and not re.fullmatch(r"https://[a-zA-Z0-9.-]+(?::\d+)?/?", config.tailscale_expected_url):
        errors.append("Expected Tailscale URL must be an HTTPS host")
    if config.remote_endpoint_url:
        parsed = urlparse(config.remote_endpoint_url)
        try:
            tailnet_ip = ipaddress.ip_address(parsed.hostname or "") in ipaddress.ip_network("100.64.0.0/10")
        except ValueError:
            tailnet_ip = False
        if parsed.scheme != "https" and not (parsed.scheme == "http" and tailnet_ip):
            errors.append("Remote endpoint requires HTTPS or an HTTP Tailscale 100.64/10 address")
        if not parsed.hostname or not parsed.path.rstrip("/").endswith("/v1") or parsed.query or parsed.fragment:
            errors.append("Remote endpoint must be an OpenAI-compatible base URL ending in /v1")
        if ";" in config.remote_endpoint_url:
            errors.append("Remote endpoint URL cannot contain a semicolon")
    if any(ch in config.remote_api_key for ch in "\r\n;"):
        errors.append("Remote API key cannot contain a newline or semicolon")
    return errors


def memory_advice(config: Config, gpus: list[GPU]) -> str:
    chosen = [gpu for gpu in gpus if gpu.uuid in config.gpu_uuids]
    total = sum(gpu.memory_mib for gpu in chosen)
    if not chosen:
        return "Select GPUs to estimate capacity."
    legacy = any(float(gpu.compute_cap) < 7.5 for gpu in chosen if re.fullmatch(r"\d+\.\d+", gpu.compute_cap))
    architecture_note = " Older GPUs may need a custom CUDA 12 build or CPU image." if legacy else ""
    model = Path(config.model_path)
    if not model.is_file():
        return f"Selected VRAM: {total / 1024:.1f} GiB. Select a GGUF to estimate fit.{architecture_note}"
    files = list(model.parent.glob("*.gguf"))
    prefix = re.sub(r"-\d{5}-of-\d{5}\.gguf$", "", model.name, flags=re.I)
    related = [p for p in files if p.name == model.name or p.name.startswith(prefix + "-")]
    size = sum(p.stat().st_size for p in related) / 2**30
    headroom = total / 1024 - size
    note = "Likely needs CPU offload; throughput may be low." if headroom < 8 else "Weight fit looks plausible; context and runtime need headroom."
    return f"Model ~{size:.1f} GiB; GPUs {total / 1024:.1f} GiB; headroom ~{headroom:.1f} GiB. {note}{architecture_note}"


def compose_document(config: Config) -> dict[str, Any]:
    model = Path(config.model_path).resolve()
    command = ["--model", f"/models/{model.name}", "--host", "0.0.0.0", "--port", "8080", "--ctx-size", str(config.context), "--split-mode", config.split_mode, "--jinja"]
    if config.gpu_layers is not None:
        command += ["--n-gpu-layers", str(config.gpu_layers)]
    if config.tensor_split:
        command += ["--tensor-split", config.tensor_split]
    mounts = [f"{model.parent}:/models:ro"]
    if config.mmproj_path:
        projector = Path(config.mmproj_path).resolve()
        if projector.parent == model.parent:
            command += ["--mmproj", f"/models/{projector.name}"]
        else:
            mounts.append(f"{projector.parent}:/projector:ro")
            command += ["--mmproj", f"/projector/{projector.name}"]
    webui_environment = {"ENABLE_OLLAMA_API": "false"}
    if config.remote_endpoint_url:
        webui_environment["OPENAI_API_BASE_URLS"] = "http://llama:8080/v1;" + config.remote_endpoint_url.rstrip("/")
    else:
        webui_environment["OPENAI_API_BASE_URL"] = "http://llama:8080/v1"
        webui_environment["OPENAI_API_KEY"] = "local"
    return {
        "services": {
            "llama": {
                "image": config.llama_image,
                "restart": "unless-stopped",
                "logging": {"driver": "json-file", "options": {"max-size": "20m", "max-file": "5"}},
                "command": command,
                "volumes": mounts,
                "deploy": {"resources": {"reservations": {"devices": [{"driver": "nvidia", "device_ids": config.gpu_uuids, "capabilities": ["gpu"]}]}}},
            },
            "open-webui": {
                "image": config.webui_image,
                "restart": "unless-stopped",
                "logging": {"driver": "json-file", "options": {"max-size": "20m", "max-file": "5"}},
                "depends_on": ["llama"],
                "ports": [f"127.0.0.1:{config.webui_port}:8080"],
                "environment": webui_environment,
                "env_file": ["./webui.env"],
                "volumes": ["webui-data:/app/backend/data"],
            },
        },
        "volumes": {"webui-data": {}},
    }


def write_plan(config: Config, gpus: list[GPU]) -> Path:
    errors = validate(config, gpus)
    if errors:
        raise ValueError("; ".join(errors))
    directory = Path(config.install_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    compose = directory / "compose.json"
    compose.write_text(json.dumps(compose_document(config), indent=2) + "\n", encoding="utf-8")
    saved = asdict(config)
    saved.pop("remote_api_key")
    (directory / "stack.json").write_text(json.dumps(saved, indent=2) + "\n", encoding="utf-8")
    secret_file = directory / "webui.env"
    existing = secret_file.read_text(encoding="utf-8") if secret_file.exists() else ""
    match = re.search(r"^WEBUI_SECRET_KEY=([a-f0-9]{64})$", existing, flags=re.M)
    secret = match.group(1) if match else secrets.token_hex(32)
    lines = [f"WEBUI_SECRET_KEY={secret}"]
    if config.remote_endpoint_url:
        lines.append(f"OPENAI_API_KEYS=local;{config.remote_api_key or 'local'}")
    secret_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if os.name == "posix":
        secret_file.chmod(0o600)
    LOG.info("Wrote deployment plan at %s", directory)
    return compose


def deploy(config: Config, gpus: list[GPU]) -> str:
    issues = check_prerequisites()
    if issues:
        raise RuntimeError("; ".join(issues))
    compose = write_plan(config, gpus)
    run(["docker", "compose", "-f", str(compose), "config", "--quiet"])
    run(["docker", "compose", "-f", str(compose), "up", "-d"], timeout=None)
    return f"Stack started. Open http://127.0.0.1:{config.webui_port}"


def save_service_logs(config: Config) -> Path:
    compose = Path(config.install_dir).expanduser().resolve() / "compose.json"
    if not compose.is_file():
        raise RuntimeError("Write a deployment plan first")
    output = run(["docker", "compose", "-f", str(compose), "logs", "--no-color", "--tail", "1000"], timeout=30).stdout
    logs = compose.parent / "logs"
    logs.mkdir(exist_ok=True)
    path = logs / "services.log"
    path.write_text(output, encoding="utf-8")
    if os.name == "posix":
        path.chmod(0o600)
    LOG.info("Saved service logs to %s", path)
    return path


def enable_tailscale(config: Config) -> str:
    if config.tailscale_mode != "serve":
        raise ValueError("Tailscale Serve is not enabled")
    if not shutil.which("tailscale"):
        raise RuntimeError("tailscale CLI is missing")
    run(["tailscale", "status"])
    run(["tailscale", "serve", "--bg", f"--https={config.tailscale_https_port}", f"http://127.0.0.1:{config.webui_port}"])
    status = run(["tailscale", "serve", "status"]).stdout.strip()
    if config.tailscale_expected_url and config.tailscale_expected_url.rstrip("/") not in status:
        LOG.warning("Expected Tailscale URL %s not in status", config.tailscale_expected_url)
        status += f"\nExpected URL {config.tailscale_expected_url} was not reported; check hostname and tailnet DNS."
    return status
