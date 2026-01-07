# ComfyUI-HY-MT1.5

ComfyUI node for Hunyuan Translation 1.5 (1.8B). Forked and adapted from the upstream project: [Tencent-Hunyuan/HY-MT](https://github.com/Tencent-Hunyuan/HY-MT.git).

<img width="875" height="846" alt="Image" src="https://github.com/user-attachments/assets/10c61cda-bcbe-4695-b04d-8fb59835da2b" />

## Install
- Place model weights under `models/HY-MT1.5/<model_dir>` (e.g. `models/HY-MT1.5/HY-MT1.5-1.8B`). The node scans this folder only. Quantized or FP16/BF16 versions are both fine as long as transformers can load them.
- Ensure `transformers` and `torch` are available (repo root already lists `transformers>=4.50.3`).
- Put this folder in `custom_nodes/ComfyUI-HY-MT1.5` (already done).
- Restart ComfyUI.

## Nodes
- `HY-MT1.5 Loader` (Category: `HY-MT1.5`): Select a subfolder inside `models/HY-MT1.5/` (e.g. `HY-MT1.5-1.8B`) and load the tokenizer/model. Outputs a reusable model handle.
- `HY-MT1.5 Translator` (Category: `HY-MT1.5`): Accepts the optional model handle from the loader (or auto-loads the first available model), target language dropdown（显示中文名称，38 个语种代码）, source text, and decoding params (`max_new_tokens`, `temperature`, `top_p`, `top_k`, `repetition_penalty`). Outputs a translated string.

## How it works
- Scans `models/HY-MT1.5/<model_dir>` and lazy-loads the tokenizer/model once (cached per path) with `torch_dtype` auto-chosen (`bfloat16` if CUDA is available).
- Builds a prompt using the official templates (Chinese template when input/target involves Chinese or Cantonese; otherwise English).
- Generates and returns only the new tokens after the prompt for a clean translation string.

## Notes
- Increase `max_new_tokens` if you need longer outputs (default 512; hard cap 4096). Longer generations cost more VRAM/time; stay within your GPU/CPU limits.
- Longer generations cost more VRAM/time; stay within your GPU/CPU limits.
