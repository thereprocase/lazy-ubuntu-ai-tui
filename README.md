# Local AI Setup TUI

An Ubuntu 26.04 terminal app for deploying a GGUF model with llama.cpp's CUDA server, Open WebUI, and optional private Tailscale Serve access. It targets general NVIDIA hosts, including mixed cards. GPU selection uses UUID, memory and compute capability; the latter distinguishes Quadro RTX 6000 (Turing) from RTX 6000 Ada and RTX PRO 6000 Blackwell.

## Requirements

- Ubuntu 26.04, Python 3.11+, NVIDIA driver and `nvidia-smi`
- Docker Engine with Compose v2 and NVIDIA Container Toolkit, configured so containers can access GPUs
- Enough **free** VRAM for model weights, KV cache and runtime overhead. Mixed GPUs may need expert tensor split tuning. System RAM allows CPU offload but can severely reduce speed.
- Tailscale installed and joined to a tailnet if private access is desired
- Hugging Face access for model browsing and downloads; log in with `hf auth login` if a repository requires it

Install from this directory:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
local-ai-tui
```

Use **Hardware** to detect and select GPUs. **Models** offers hardware-aware starting points and searches Hugging Face GGUF repos, displays quantization groups and sizes, downloads the selected group, or accepts a local GGUF path. For split models, link the first shard. If using image input, link the matching local `mmproj` file as well. Presets fill in the repository, quant, context and backend; select the intended GPUs separately.

The presets include Qwen3.8 27B Q8 for two GPUs, Q5 for a 24 GB card, Q3 for a 16 GB card, Qwen3.5 9B Q8 as a faster 16 GB option, Kimi Linear 48B A3B Q4 for two GPUs, and Kimi K3 1-bit or 2-bit for large-RAM hosts. File size is only a starting estimate; leave VRAM for context and runtime. The Kimi K3 choices require roughly 610 GB or 880 GB of total memory respectively, and the TUI checks free disk space before downloading. K3 will offload most weights to system RAM on a 40 GB GPU pair and may generate slowly.

Use **Stack** to write and validate `compose.json`, then deploy. Open WebUI binds to `127.0.0.1` on the configured port. It uses a named volume for data and a persistent secret in `webui.env`. The first Open WebUI user becomes the administrator. Use **Tailscale** to expose that local port privately via Serve. The expected URL field checks the resulting node URL; Tailscale controls the actual hostname.

The Tailscale tab can also add a remote OpenAI-compatible `/v1` endpoint to Open WebUI. Supply an HTTPS endpoint or an HTTP `100.64.0.0/10` tailnet address. Its optional API key is stored only in the mode-0600 `webui.env` file, not in `stack.json` or the setup log. If Docker cannot resolve a MagicDNS name, use the remote node's 100.x address.

Open WebUI persists connection settings in its data volume. After the first launch, changes to endpoint fields may need to be made in Open WebUI's **Admin > Connections** page as well.

Simple mode shows the primary choices. Advanced exposes context, port, backend and Tailscale options. Expert exposes GPU layer count, split mode, tensor ratios and image tags. The default context is 65,536 tokens, but actual fit depends on the model, quant, GPU split and runtime buffers. GPU layer count is blank by default so llama.cpp can fit the model to available VRAM. Settings are written to `stack.json` in the deployment directory. The setup log rotates at `~/.local/state/local-ai-tui/setup.log` or `$XDG_STATE_HOME/local-ai-tui/setup.log`. Container logs use Docker's rotating JSON file driver; **Save service logs** exports the latest 1,000 lines per service to `logs/services.log`.

The default model link is `ggml-org/Qwen3.8-27B-GGUF` with `Qwen3.8-27B-Q8_0.gguf`, but any supported GGUF repository or local file can be used. The default llama.cpp image is CUDA 12; experts can change it for a supported CUDA generation. For very new GGUF architectures, use a recent llama.cpp image. The model browser reports repository file sizes, while the memory estimate checks local model shards and should be treated as an estimate.

Kimi K3 uses a separate locally built image from [Unsloth's K3 llama.cpp fork](https://unsloth.ai/docs/models/kimi-k3). The generated `Dockerfile.kimi-k3` pins the fork revision and compiles for the selected GPUs' compute capabilities. The first build downloads source and CUDA build dependencies. Its `--cpu-moe` and `--fit on` settings favor RAM offload; all GGUF shards must be present before deployment. The TUI downloads the matching vision projector with K3. K3 is an experimental path; it has not been tested on every NVIDIA architecture or with every Open WebUI feature. See the [K3 model and quantizations](https://huggingface.co/unsloth/Kimi-K3-GGUF) for current requirements.

NVIDIA's [compute capability table](https://developer.nvidia.com/cuda/gpus) distinguishes the Quadro RTX 6000 (7.5), RTX A4000 (8.6), RTX 6000 Ada (8.9), and RTX PRO 6000 Blackwell (12.0). CUDA 12.8 supports Blackwell, but older Volta, Pascal and Maxwell cards may need a compatible custom llama.cpp image and driver; the TUI flags them during selection.

## Useful commands after deployment

```bash
docker compose -f ~/local-ai-stack/compose.json ps
docker compose -f ~/local-ai-stack/compose.json logs -f llama
docker compose -f ~/local-ai-stack/compose.json logs -f open-webui
tailscale serve status
```

The app does not install system drivers, Docker, NVIDIA Container Toolkit, or Tailscale. It reports missing prerequisites before deployment. It does not use Tailscale Funnel or expose Open WebUI directly on a public interface.
