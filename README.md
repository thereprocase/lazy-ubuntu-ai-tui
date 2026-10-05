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

Use **Hardware** to detect and select GPUs. **Models** searches Hugging Face GGUF repos, displays quantization groups and sizes, downloads the selected group, or accepts a local GGUF path. For split models, link the first shard. If using image input, link the matching local `mmproj` file as well.

Use **Stack** to write and validate `compose.json`, then deploy. Open WebUI binds to `127.0.0.1` on the configured port. It uses a named volume for data and a persistent secret in `webui.env`. The first Open WebUI user becomes the administrator. Use **Tailscale** to expose that local port privately via Serve. The expected URL field checks the resulting node URL; Tailscale controls the actual hostname.

The Tailscale tab can also add a remote OpenAI-compatible `/v1` endpoint to Open WebUI. Supply an HTTPS endpoint or an HTTP `100.64.0.0/10` tailnet address. Its optional API key is stored only in the mode-0600 `webui.env` file, not in `stack.json` or the setup log. If Docker cannot resolve a MagicDNS name, use the remote node's 100.x address.

Open WebUI persists connection settings in its data volume. After the first launch, changes to endpoint fields may need to be made in Open WebUI's **Admin > Connections** page as well.

Simple mode shows the primary choices. Advanced exposes context, port and Tailscale options. Expert exposes GPU layer count, split mode, tensor ratios and image tags. GPU layer count is blank by default so llama.cpp can fit the model to available VRAM. Settings are written to `stack.json` in the deployment directory. The setup log rotates at `~/.local/state/local-ai-tui/setup.log` or `$XDG_STATE_HOME/local-ai-tui/setup.log`. Container logs use Docker's rotating JSON file driver; **Save service logs** exports the latest 1,000 lines per service to `logs/services.log`.

The default model link is `unsloth/Qwen3.8-Flash-Next-GGUF` with `UD-Q4_K_XL`, but any GGUF repository or local file can be used. The default llama.cpp image is CUDA 12; experts can change it for a supported CUDA generation. For very new GGUF architectures, use a recent llama.cpp image. The model browser reports repository file sizes, while the memory estimate checks local model shards and should be treated as an estimate.

NVIDIA's [compute capability table](https://developer.nvidia.com/cuda/gpus) distinguishes the Quadro RTX 6000 (7.5), RTX A4000 (8.6), RTX 6000 Ada (8.9), and RTX PRO 6000 Blackwell (12.0). CUDA 12.8 supports Blackwell, but older Volta, Pascal and Maxwell cards may need a compatible custom llama.cpp image and driver; the TUI flags them during selection.

## Useful commands after deployment

```bash
docker compose -f ~/local-ai-stack/compose.json ps
docker compose -f ~/local-ai-stack/compose.json logs -f llama
docker compose -f ~/local-ai-stack/compose.json logs -f open-webui
tailscale serve status
```

The app does not install system drivers, Docker, NVIDIA Container Toolkit, or Tailscale. It reports missing prerequisites before deployment. It does not use Tailscale Funnel or expose Open WebUI directly on a public interface.
