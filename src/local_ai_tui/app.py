from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
import re
import shutil

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Checkbox, DataTable, Footer, Header, Input, Label, RichLog, Select, Static, TabbedContent, TabPane

from .core import Config, DEFAULT_QUANT, DEFAULT_REPO, GPU, K3_BACKEND, K3_REPO, MODEL_PRESETS, check_prerequisites, configure_logging, deploy, detect_gpus, enable_tailscale, memory_advice, save_service_logs, write_plan

LOG = logging.getLogger("local_ai_tui")


class LocalAIApp(App):
    TITLE = "Local AI Setup"
    SUB_TITLE = "NVIDIA GGUF · llama.cpp · Open WebUI · Tailscale"
    CSS = """
    Screen { background: $surface; }
    VerticalScroll { padding: 1 2; }
    .field { margin: 0 0 1 0; }
    .hint { color: $text-muted; margin: 0 0 1 0; }
    .row { height: auto; margin-bottom: 1; }
    Button { margin-right: 1; }
    DataTable { height: 12; margin-bottom: 1; }
    #gpu-list { height: auto; min-height: 4; }
    #messages { height: 10; border: round $accent; margin: 1 2; }
    #memory { margin: 1 0; color: $warning; }
    """

    def __init__(self) -> None:
        super().__init__()
        self.config = Config()
        self.gpus: list[GPU] = []
        self.quant_files: dict[str, list[str]] = {}
        self.quant_sizes: dict[str, int] = {}
        self.log_file = configure_logging()

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent():
            with TabPane("Hardware", id="hardware"):
                with VerticalScroll():
                    yield Label("Complexity")
                    yield Select((("Simple", "simple"), ("Advanced", "advanced"), ("Expert", "expert")), value="simple", id="complexity", classes="field")
                    yield Static("Detects NVIDIA GPUs by UUID, VRAM and compute capability. Check the exact model before selecting cards with similar names.", classes="hint")
                    yield Button("Detect GPUs", id="detect")
                    yield Vertical(id="gpu-list")
                    yield Static(id="memory")
                    yield Button("Check prerequisites", id="check")
            with TabPane("Models", id="models"):
                with VerticalScroll():
                    yield Static("Search Hugging Face GGUF repositories, inspect their quantizations, download a selection, or link an existing local file.", classes="hint")
                    yield Label("Recommended starting points")
                    yield Select(tuple((item[0], key) for key, item in MODEL_PRESETS.items()), value="qwen-dual", id="preset", classes="field")
                    yield Button("Apply recommendation", id="apply-preset")
                    yield Static("Kimi K3 is a RAM-offload experiment: 594 GB to 861 GB download, Unsloth fork build, and slow generation on small GPUs.", classes="hint")
                    yield Input(value="Qwen3.8-27B", placeholder="Search GGUF models", id="search", classes="field")
                    yield Button("Search", id="search-button")
                    yield DataTable(id="repos")
                    yield Input(value=DEFAULT_REPO, placeholder="owner/repository", id="repo", classes="field")
                    yield Button("Inspect repository", id="inspect")
                    yield DataTable(id="quants")
                    yield Input(value=DEFAULT_QUANT, placeholder="Quant folder or filename", id="quant", classes="field")
                    yield Button("Download selected quant", id="download")
                    yield Input(placeholder="Absolute path to local .gguf (first shard)", id="model-path", classes="field")
                    yield Input(placeholder="Optional local mmproj path for vision", id="mmproj-path", classes="field")
                    yield Static(id="model-estimate")
            with TabPane("Stack", id="stack"):
                with VerticalScroll():
                    yield Input(value=self.config.install_dir, placeholder="Deployment directory", id="install-dir", classes="field")
                    yield Input(value="65536", placeholder="Context tokens", id="context", classes="field advanced")
                    yield Input(value="3000", placeholder="Open WebUI localhost port", id="webui-port", classes="field advanced")
                    yield Select((("Standard llama.cpp CUDA", "standard"), ("Unsloth Kimi K3 fork (build locally)", K3_BACKEND)), value="standard", id="backend", classes="field advanced")
                    yield Select((("Layer", "layer"), ("Row", "row"), ("None", "none")), value="layer", id="split-mode", classes="field expert")
                    yield Input(placeholder="GPU layers (blank = automatic fit)", id="gpu-layers", classes="field expert")
                    yield Input(placeholder="Tensor split ratios, e.g. 3,2,1", id="tensor-split", classes="field expert")
                    yield Input(value=self.config.llama_image, placeholder="llama.cpp image", id="llama-image", classes="field expert")
                    yield Input(value=self.config.webui_image, placeholder="Open WebUI image", id="webui-image", classes="field expert")
                    with Horizontal(classes="row"):
                        yield Button("Write & validate plan", id="plan")
                        yield Button("Deploy stack", id="deploy", variant="primary")
                        yield Button("Save service logs", id="service-logs")
                    yield Static("Open WebUI binds to 127.0.0.1. Docker Compose requires the NVIDIA Container Toolkit. Deployment pulls images and creates containers.", classes="hint")
            with TabPane("Tailscale", id="tailscale"):
                with VerticalScroll():
                    yield Select((("Off", "off"), ("Private Serve", "serve")), value="off", id="tailscale-mode", classes="field")
                    yield Input(value="443", placeholder="Tailscale HTTPS port", id="tailscale-port", classes="field advanced")
                    yield Input(placeholder="Expected tailnet URL, e.g. https://host.example.ts.net", id="tailscale-url", classes="field advanced")
                    yield Static("Private Serve proxies the local Open WebUI port to this node's tailnet URL. The hostname comes from Tailscale; the expected URL is checked after setup.", classes="hint")
                    yield Button("Enable private endpoint", id="tailscale-enable")
                    yield Static("Optional: connect Open WebUI to another OpenAI-compatible endpoint on your tailnet. Use its full /v1 base URL.", classes="hint")
                    yield Input(placeholder="https://model-node.example.ts.net/v1 or http://100.x.y.z:port/v1", id="remote-url", classes="field advanced")
                    yield Input(placeholder="Optional remote API key", password=True, id="remote-key", classes="field advanced")
        yield RichLog(id="messages", wrap=True, highlight=True, markup=True)
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#repos", DataTable).add_columns("Repository", "Downloads")
        self.query_one("#quants", DataTable).add_columns("Quant / file group", "Size GiB", "Files")
        self._complexity()
        self._say(f"Setup log: {self.log_file}")
        self.detect_worker()

    def _say(self, message: str, error: bool = False) -> None:
        LOG.error(message) if error else LOG.info(message)
        color = "red" if error else "green"
        self.query_one("#messages", RichLog).write(f"[{color}]{message}[/]")

    def _complexity(self) -> None:
        level = self.query_one("#complexity", Select).value
        for widget in self.query(".advanced"):
            widget.display = level in {"advanced", "expert"}
        for widget in self.query(".expert"):
            widget.display = level == "expert"

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "complexity":
            self._complexity()

    def _apply_preset(self) -> None:
        key = str(self.query_one("#preset", Select).value)
        label, repo, quant, context, backend, memory_gb = MODEL_PRESETS[key]
        self.query_one("#repo", Input).value = repo
        self.query_one("#quant", Input).value = quant
        self.query_one("#context", Input).value = str(context)
        self.query_one("#backend", Select).value = backend
        self.query_one("#model-path", Input).value = ""
        self.query_one("#mmproj-path", Input).value = ""
        self._update_memory()
        self._say(f"Selected {label}. Inspect repository, select its GGUF group, then download or link a local file. Approximate memory requirement: {memory_gb} GB plus runtime overhead.")
        self.inspect_worker(repo)

    async def _render_gpus(self) -> None:
        container = self.query_one("#gpu-list", Vertical)
        await container.remove_children()
        if not self.gpus:
            await container.mount(Static("No NVIDIA GPUs detected. Run this TUI on the target Ubuntu host."))
            return
        await container.mount(Static("Select GPUs for this model:"))
        for gpu in self.gpus:
            label = f"[{gpu.index}] {gpu.name} · {gpu.memory_mib / 1024:.1f} GiB · CC {gpu.compute_cap} {gpu.architecture} · {gpu.uuid}"
            await container.mount(Checkbox(label, value=True, id=f"gpu-{gpu.index}"))
        self._update_memory()

    def _selected_gpus(self) -> list[str]:
        return [gpu.uuid for gpu in self.gpus if self.query(f"#gpu-{gpu.index}") and self.query_one(f"#gpu-{gpu.index}", Checkbox).value]

    def _update_memory(self) -> None:
        self.config.gpu_uuids = self._selected_gpus() if self.gpus else []
        self.config.model_path = self.query_one("#model-path", Input).value.strip()
        self.query_one("#memory", Static).update(memory_advice(self.config, self.gpus))

    def on_checkbox_changed(self, _: Checkbox.Changed) -> None:
        self._update_memory()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "model-path":
            self._update_memory()

    def _read_config(self) -> Config:
        def value(widget_id: str) -> str:
            return self.query_one(f"#{widget_id}", Input).value.strip()
        self.config.complexity = str(self.query_one("#complexity", Select).value)
        self.config.install_dir = value("install-dir")
        self.config.model_path = value("model-path")
        self.config.mmproj_path = value("mmproj-path")
        self.config.repo_id = value("repo")
        self.config.quant = value("quant")
        self.config.gpu_uuids = self._selected_gpus() if self.gpus else []
        self.config.context = int(value("context"))
        self.config.webui_port = int(value("webui-port"))
        self.config.split_mode = str(self.query_one("#split-mode", Select).value)
        self.config.tensor_split = value("tensor-split")
        self.config.gpu_layers = int(value("gpu-layers")) if value("gpu-layers") else None
        self.config.llama_image = value("llama-image")
        self.config.backend = str(self.query_one("#backend", Select).value)
        self.config.webui_image = value("webui-image")
        self.config.tailscale_mode = str(self.query_one("#tailscale-mode", Select).value)
        self.config.tailscale_https_port = int(value("tailscale-port"))
        self.config.tailscale_expected_url = value("tailscale-url")
        self.config.remote_endpoint_url = value("remote-url")
        self.config.remote_api_key = value("remote-key")
        return Config(**asdict(self.config))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button = event.button.id
        try:
            if button == "detect":
                self.detect_worker()
            elif button == "check":
                self.check_worker()
            elif button == "search-button":
                self.search_worker(self.query_one("#search", Input).value.strip())
            elif button == "inspect":
                self.inspect_worker(self.query_one("#repo", Input).value.strip())
            elif button == "apply-preset":
                self._apply_preset()
            elif button == "download":
                self.download_worker(self.query_one("#repo", Input).value.strip(), self.query_one("#quant", Input).value.strip(), self.query_one("#install-dir", Input).value.strip())
            elif button == "plan":
                path = write_plan(self._read_config(), self.gpus)
                self._say(f"Plan written: {path}")
            elif button == "deploy":
                self.deploy_worker(self._read_config())
            elif button == "service-logs":
                self.logs_worker(self._read_config())
            elif button == "tailscale-enable":
                self.tailscale_worker(self._read_config())
        except (ValueError, RuntimeError, OSError) as exc:
            self._say(str(exc), error=True)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id == "repos":
            self.query_one("#repo", Input).value = str(event.row_key.value)
            self.inspect_worker(str(event.row_key.value))
        elif event.data_table.id == "quants":
            self.query_one("#quant", Input).value = str(event.row_key.value)

    @work(thread=True)
    def detect_worker(self) -> None:
        found = detect_gpus()
        self.call_from_thread(self._detected, found)

    def _detected(self, found: list[GPU]) -> None:
        self.gpus = found
        self.run_worker(self._render_gpus(), group="gpu-render", exclusive=True)
        self._say(f"Detected {len(found)} NVIDIA GPU(s).")

    @work(thread=True)
    def check_worker(self) -> None:
        issues = check_prerequisites()
        self.call_from_thread(self._say, "; ".join(issues) if issues else "Prerequisites are available.", bool(issues))

    @work(thread=True)
    def search_worker(self, term: str) -> None:
        try:
            from huggingface_hub import HfApi
            if not term:
                raise ValueError("Enter a search term")
            models = list(HfApi().list_models(search=term, filter="gguf", sort="downloads", limit=30))
            rows = [(model.id, str(model.downloads or 0)) for model in models]
            self.call_from_thread(self._show_repos, rows)
        except Exception as exc:
            LOG.exception("Model search failed")
            self.call_from_thread(self._say, f"Model search failed: {exc}", True)

    def _show_repos(self, rows: list[tuple[str, str]]) -> None:
        table = self.query_one("#repos", DataTable)
        table.clear()
        for repo, downloads in rows:
            table.add_row(repo, downloads, key=repo)
        self._say(f"Found {len(rows)} GGUF repositories.")

    @work(thread=True)
    def inspect_worker(self, repo: str) -> None:
        try:
            from huggingface_hub import HfApi
            if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
                raise ValueError("Enter a Hugging Face owner/repository")
            grouped: dict[str, list[tuple[str, int]]] = {}
            for item in HfApi().list_repo_tree(repo, recursive=True):
                if not hasattr(item, "size") or not item.path.lower().endswith(".gguf") or "mmproj" in item.path.lower():
                    continue
                folder = item.path.rsplit("/", 1)[0] if "/" in item.path else ""
                name = item.path.rsplit("/", 1)[-1]
                quant = folder or re.sub(r"-\d{5}-of-\d{5}\.gguf$", "", name, flags=re.I)
                grouped.setdefault(quant, []).append((item.path, item.size or 0))
            self.call_from_thread(self._show_quants, grouped)
        except Exception as exc:
            LOG.exception("Repository inspection failed")
            self.call_from_thread(self._say, f"Repository inspection failed: {exc}", True)

    def _show_quants(self, grouped: dict[str, list[tuple[str, int]]]) -> None:
        table = self.query_one("#quants", DataTable)
        table.clear()
        self.quant_files = {group: [path for path, _ in files] for group, files in grouped.items()}
        self.quant_sizes = {group: sum(size for _, size in files) for group, files in grouped.items()}
        for group, files in sorted(grouped.items()):
            table.add_row(group, f"{sum(size for _, size in files) / 2**30:.1f}", str(len(files)), key=group)
        self._say(f"Found {len(grouped)} GGUF groups. Select one to download.")

    @work(thread=True)
    def download_worker(self, repo: str, quant: str, install_dir: str) -> None:
        try:
            from huggingface_hub import snapshot_download
            files = self.quant_files.get(quant)
            if not files:
                raise ValueError("Inspect the repository and select a quant first")
            target = Path(install_dir).expanduser() / "models" / repo.replace("/", "--")
            target.mkdir(parents=True, exist_ok=True)
            required = self.quant_sizes.get(quant, 0)
            if required and shutil.disk_usage(target).free < required * 1.05:
                raise RuntimeError(f"Not enough free disk space for {required / 2**30:.1f} GiB quant")
            if repo == K3_REPO:
                files = [*files, "mmproj-BF16.gguf"]
            self.call_from_thread(self._say, f"Downloading {len(files)} GGUF file(s) to {target} …")
            snapshot_download(repo_id=repo, allow_patterns=files, local_dir=target)
            first = sorted(path for path in files if path.lower().endswith(".gguf") and "mmproj" not in path.lower())[0]
            self.call_from_thread(self._downloaded, str(target / first), str(target / "mmproj-BF16.gguf") if repo == K3_REPO else "")
        except Exception as exc:
            LOG.exception("Model download failed")
            self.call_from_thread(self._say, f"Download failed: {exc}", True)

    def _downloaded(self, path: str, mmproj: str = "") -> None:
        self.query_one("#model-path", Input).value = path
        if mmproj:
            self.query_one("#mmproj-path", Input).value = mmproj
        self._update_memory()
        self._say(f"Model linked: {path}")

    @work(thread=True)
    def deploy_worker(self, config: Config) -> None:
        try:
            self.call_from_thread(self._say, "Validating and starting Docker Compose …")
            result = deploy(config, self.gpus)
            self.call_from_thread(self._say, result)
        except Exception as exc:
            LOG.exception("Deployment failed")
            self.call_from_thread(self._say, f"Deployment failed: {exc}", True)

    @work(thread=True)
    def logs_worker(self, config: Config) -> None:
        try:
            path = save_service_logs(config)
            self.call_from_thread(self._say, f"Service logs saved: {path}")
        except Exception as exc:
            LOG.exception("Saving service logs failed")
            self.call_from_thread(self._say, f"Saving logs failed: {exc}", True)

    @work(thread=True)
    def tailscale_worker(self, config: Config) -> None:
        try:
            result = enable_tailscale(config)
            self.call_from_thread(self._say, result)
        except Exception as exc:
            LOG.exception("Tailscale setup failed")
            self.call_from_thread(self._say, f"Tailscale failed: {exc}", True)


def main() -> None:
    LocalAIApp().run()


if __name__ == "__main__":
    main()
