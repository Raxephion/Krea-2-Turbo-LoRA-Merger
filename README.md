# Krea 2 Turbo LoRA Merger

A standalone, offline Gradio app that permanently merges one or more LoRA
files into a base diffusion model checkpoint (built for Krea 2 Turbo, but
works with any safetensors checkpoint using compatible LoRA naming).

!!! APP IS IN BETA!!!

No internet access is required to run it — everything happens locally on
your machine.

## Requirements

- Windows, with Python 3.10 or 3.11 installed and on PATH
  (https://www.python.org/downloads/ — check "Add python.exe to PATH")
- Your base checkpoint and LoRA file(s) already downloaded locally as
  `.safetensors` files

## Install

Double-click `install.bat` (or run it from a terminal). It will:

1. Create a local virtual environment in `.venv`
2. Install Gradio, PyTorch (CPU version), and safetensors

If you have an NVIDIA GPU and want faster merging, after `install.bat`
finishes, run:

```
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

(swap `cu121` for the CUDA build matching your GPU driver — see
https://pytorch.org/get-started/locally/)

## Run

Double-click `run.bat`. This opens the app in your browser at
`http://127.0.0.1:7860`.

## Using the app

1. **Load files** — upload your base model `.safetensors` file, and one or
   more LoRA `.safetensors` files.
2. **LoRA weights** — a weight slider appears for each LoRA you loaded.
   `1.0` applies the LoRA at its trained strength; lower values apply it
   more subtly, negative values invert its effect.
3. **Settings**:
   - **Compute precision** — precision used for the merge math itself.
     `fp32` is safest/most accurate; `fp16`/`bf16` are faster and use less
     RAM but can lose a little precision.
   - **Output precision** — precision of the saved file. `keep original`
     preserves whatever precision the base checkpoint was in.
   - **Device** — `cpu` works everywhere; `cuda` is faster if you have a
     working GPU build of PyTorch installed.
   - **Output folder / filename** — where the merged `.safetensors` file
     is written.
4. **Check compatibility (dry run)** — before merging, run this to see how
   many LoRA layers actually match your base model's weight names. LoRAs
   trained for a different architecture, or exported in an unexpected
   format, will show up here as "unmatched" rather than silently failing.
5. **Merge & Save** — performs the actual merge and writes the output file.
   The log shows how many layers were merged, skipped, or mismatched.

## How it works

LoRA weight files store, for a subset of layers, a low-rank pair of
matrices (`down`/`up`, sometimes named `lora_A`/`lora_B`) plus an optional
`alpha` scaling value. Merging bakes each LoRA in by computing, for every
matched layer:

```
new_weight = base_weight + (up @ down) * (alpha / rank) * your_weight_slider
```

This is applied directly to the base checkpoint's tensors and the result
is saved as a new, standalone `.safetensors` file — no LoRA loader needed
at inference time.

The app recognizes both common LoRA naming conventions (Kohya/ComfyUI-style
`lora_down`/`lora_up`, and Diffusers/PEFT-style `lora_A`/`lora_B`), and
falls back to fuzzy key matching when prefixes differ between the LoRA and
base file. Any layer it can't confidently match is reported, not guessed.

## Notes / limitations

- This performs a **linear-layer LoRA merge**. It assumes standard
  `(out_features, in_features)` weight/LoRA shapes, which covers Krea 2's
  transformer architecture (and most modern DiT/UNet models). It is not
  designed for convolutional LoRAs.
- Merging loads the full base model into memory (RAM, or VRAM if using
  `cuda`). Make sure you have enough — check your checkpoint's file size
  as a rough guide.
- Always run "Check compatibility" first on a new LoRA/base combination
  before trusting the merged output.
