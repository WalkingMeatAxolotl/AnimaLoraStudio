# Windows ROCm guide

This branch supports Windows ROCm while retaining PyTorch's `torch.cuda` API and
`cuda` device string. It detects AMD builds through `torch.version.hip`, uses
bf16 PyTorch SDPA, and rejects CUDA-only xformers/flash-attn backends.

Use `studio_rocm.bat --check` for preflight and `studio_rocm.bat` for Studio.
The default interpreter is `E:\aiwork\python_embeded\python.exe`; override it
with `ANIMA_ROCM_PYTHON`. For CLI training, copy
[`examples/rocm/anima-windows-smoke.example.yaml`](../../examples/rocm/anima-windows-smoke.example.yaml)
and run `train_rocm.bat path\to\config.yaml`.

ComfyUI single-file Anima Qwen weights are supported. The loader discovers the
Qwen and T5 tokenizers from `ComfyUI/comfy/text_encoders`, so weights under
`ComfyUI/models/clip` do not need to be converted into a Hugging Face directory.

The validated path is Python 3.12, PyTorch 2.9.1 + ROCm/HIP 7.14, gfx1100,
Anima 2B, bf16, and SDPA. Krea 2 training on Windows ROCm is not yet claimed as
validated. Do not commit weights, datasets, caches, or training output.

The ROCm launchers redirect MIOpen's writable database and kernel cache to the
ignored `.cache\miopen` directory. Set `ANIMA_ROCM_CACHE_DIR` to override it.
Direct `runtime\anima_train.py` execution applies the same default before torch
is imported.

Set `latent_cache_dir` in the training YAML to keep `.npz` files in a separate
writable location when the source dataset is read-only. Leaving it blank keeps
the upstream behavior of storing each cache next to its image.

The example starts at 256 px only to validate one complete
forward/backward/save cycle. Increase to 512/1024 after it passes; portable
SDPA at high resolutions can be much slower than CUDA flash-attn. The smoke
config enables `kv_trim` and disables gradient checkpointing at 256 px; turn
checkpointing back on when higher resolutions need the VRAM savings.
