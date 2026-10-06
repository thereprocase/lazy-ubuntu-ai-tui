import json
from pathlib import Path
import shutil
import subprocess

import pytest

from local_ai_tui.core import Config, GPU, K3_BACKEND, K3_REPO, MODEL_PRESETS, compose_document, memory_advice, validate, write_plan


def test_architecture_uses_compute_capability_not_product_name():
    assert GPU(0, "a", "Quadro RTX 6000", 24576, "7.5").architecture == "Turing"
    assert GPU(1, "b", "RTX 6000 Ada", 49152, "8.9").architecture == "Ada"
    assert GPU(2, "c", "RTX PRO 6000", 98304, "12.0").architecture == "Blackwell"
    assert GPU(3, "d", "RTX A4000", 16384, "8.6").architecture == "Ampere"
    assert GPU(4, "e", "B200", 196608, "10.0").architecture == "Blackwell"


def test_plan_targets_selected_gpu_and_local_only_webui(tmp_path: Path):
    model = tmp_path / "model-00001-of-00002.gguf"
    model.write_bytes(b"x" * 100)
    (tmp_path / "model-00002-of-00002.gguf").write_bytes(b"y" * 100)
    gpu = GPU(0, "GPU-abc", "RTX A4000", 16384, "8.6")
    config = Config(install_dir=str(tmp_path / "stack"), model_path=str(model), gpu_uuids=[gpu.uuid])
    assert validate(config, [gpu]) == []
    compose = compose_document(config)
    assert compose["services"]["llama"]["deploy"]["resources"]["reservations"]["devices"][0]["device_ids"] == ["GPU-abc"]
    assert compose["services"]["open-webui"]["ports"] == ["127.0.0.1:3000:8080"]
    assert compose["services"]["open-webui"]["environment"]["OPENAI_API_BASE_URL"] == "http://llama:8080/v1"
    assert "Model ~0.0 GiB" in memory_advice(config, [gpu])
    plan = write_plan(config, [gpu])
    assert json.loads(plan.read_text())["services"]["llama"]["command"][1] == "/models/model-00001-of-00002.gguf"
    assert (plan.parent / "webui.env").exists()


def test_invalid_split_and_missing_gpu(tmp_path: Path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"x")
    gpu = GPU(0, "GPU-abc", "GPU", 16384, "8.6")
    config = Config(model_path=str(model), gpu_uuids=["GPU-missing"], tensor_split="1,2")
    errors = validate(config, [gpu])
    assert any("no longer present" in error for error in errors)
    assert any("one value" in error for error in errors)


def test_generated_compose_is_accepted_by_docker(tmp_path: Path):
    if not shutil.which("docker"):
        pytest.skip("Docker CLI unavailable")
    model = tmp_path / "model.gguf"
    model.write_bytes(b"x")
    gpu = GPU(0, "GPU-abc", "GPU", 16384, "8.6")
    plan = write_plan(Config(install_dir=str(tmp_path / "stack"), model_path=str(model), gpu_uuids=[gpu.uuid]), [gpu])
    result = subprocess.run(["docker", "compose", "-f", str(plan), "config", "--quiet"], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_remote_tailnet_connection_keeps_key_out_of_plan(tmp_path: Path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"x")
    gpu = GPU(0, "GPU-abc", "GPU", 16384, "8.6")
    config = Config(
        install_dir=str(tmp_path / "stack"), model_path=str(model), gpu_uuids=[gpu.uuid],
        remote_endpoint_url="https://models.example.ts.net/v1", remote_api_key="secret-token",
    )
    assert validate(config, [gpu]) == []
    plan = write_plan(config, [gpu])
    assert "secret-token" not in plan.read_text()
    assert "secret-token" not in (plan.parent / "stack.json").read_text()
    assert "OPENAI_API_KEYS=local;secret-token" in (plan.parent / "webui.env").read_text()
    assert compose_document(config)["services"]["open-webui"]["environment"]["OPENAI_API_BASE_URLS"].endswith(";https://models.example.ts.net/v1")


def test_k3_plan_uses_pinned_fork_and_cpu_offload(tmp_path: Path):
    model = tmp_path / "Kimi-K3-UD-IQ1_S-00001-of-00002.gguf"
    model.write_bytes(b"x")
    gpu = GPU(0, "GPU-abc", "Quadro RTX 6000", 24576, "7.5")
    config = Config(install_dir=str(tmp_path / "stack"), model_path=str(model), repo_id=K3_REPO, quant="UD-IQ1_S", backend=K3_BACKEND, gpu_uuids=[gpu.uuid])
    assert any("Missing 1 GGUF shard" in error for error in validate(config, [gpu]))
    (tmp_path / "Kimi-K3-UD-IQ1_S-00002-of-00002.gguf").write_bytes(b"y")
    assert validate(config, [gpu]) == []
    plan = write_plan(config, [gpu])
    document = json.loads(plan.read_text())
    llama = document["services"]["llama"]
    assert llama["build"]["args"]["CUDA_ARCHITECTURES"] == "75"
    assert "--cpu-moe" in llama["command"]
    assert "768d2a481a99cb75ec9a03b95dadbd35e7acf496" in (plan.parent / "Dockerfile.kimi-k3").read_text()
    assert MODEL_PRESETS["kimi-k3-1bit"][1] == K3_REPO
    if shutil.which("docker"):
        result = subprocess.run(["docker", "compose", "-f", str(plan), "config", "--quiet"], text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
