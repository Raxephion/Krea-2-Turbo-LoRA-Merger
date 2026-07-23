"""
Krea 2 Turbo LoRA Merger
------------------------
Standalone, offline Gradio app that permanently merges one or more LoRA
files into a base diffusion model checkpoint (designed for Krea 2 Turbo,
but works on any safetensors DiT/UNet checkpoint with compatible LoRA
naming conventions).

Run with:  python app.py
"""

from __future__ import annotations

import os
import traceback
from datetime import datetime

import gradio as gr
import torch

from lora_merge import (
    DTYPE_MAP,
    check_compatibility,
    load_state_dict,
    merge_lora_file,
    save_merged,
)

MAX_LORAS = 8
COMPUTE_DTYPES = ["fp32", "fp16", "bf16"]
OUTPUT_DTYPES = list(DTYPE_MAP.keys())

# Gradio moved the `css` argument from Blocks() to launch() in v6.0.
# Passing it to the wrong one either errors or gets silently ignored,
# depending on version, so detect which API this install uses.
try:
    _GRADIO_MAJOR = int(gr.__version__.split(".")[0])
except Exception:
    _GRADIO_MAJOR = 0  # unknown -> assume older (pre-6.0) API as the safer default

CYBERPUNK_CSS = """
/* --- CYBERPUNK HUD STYLES --- */
:root {
    --neon-blue: #00ffff;
    --neon-glow: 0 0 8px #00ffff, 0 0 16px #00ffff;
    --bg-black: #000000;
}
/* GLOBAL BACKGROUND */
body, .gradio-container {
    background: var(--bg-black) !important;
    color: var(--neon-blue) !important;
    font-family: 'Share Tech Mono', monospace !important;
}
/* MAIN TITLE */
#main_title {
    color: var(--neon-blue) !important;
    text-shadow: var(--neon-glow);
    text-align: center;
    font-size: 3em !important;
    border-bottom: 1px solid var(--neon-blue);
    padding-bottom: 8px;
}
/* TEXTBOXES */
textarea, input[type="text"], .gr-textbox, .gr-input {
    background: var(--bg-black) !important;
    border: 1px solid var(--neon-blue) !important;
    color: var(--neon-blue) !important;
    text-shadow: var(--neon-glow);
    font-family: 'Share Tech Mono', monospace !important;
    border-radius: 0 !important;
    box-shadow: inset 0 0 12px #003333;
}
/* OUTPUT + INPUT TEXTAREAS */
#danika_output textarea, #user_input textarea {
    background: var(--bg-black) !important;
    color: var(--neon-blue) !important;
    border: 1px solid var(--neon-blue) !important;
    text-shadow: var(--neon-glow);
    font-size: 15px !important;
    box-shadow: inset 0 0 20px #002222;
}
/* BUTTONS */
button, .gr-button {
    background: var(--bg-black) !important;
    border: 1px solid var(--neon-blue) !important;
    color: var(--neon-blue) !important;
    text-shadow: var(--neon-glow);
    border-radius: 0 !important;
    transition: all 0.2s ease-in-out;
}
button:hover, .gr-button:hover {
    background: var(--neon-blue) !important;
    color: var(--bg-black) !important;
    text-shadow: none !important;
    box-shadow: 0 0 20px #00ffff;
}
/* ACCORDIONS & BOXES */
.gr-accordion, .gr-box {
    background: var(--bg-black) !important;
    border: 1px solid var(--neon-blue) !important;
    color: var(--neon-blue) !important;
    text-shadow: var(--neon-glow);
    border-radius: 0 !important;
}
/* SLIDERS */
input[type=range]::-webkit-slider-thumb {
    background: var(--neon-blue) !important;
    box-shadow: var(--neon-glow);
}
input[type=range]::-webkit-slider-runnable-track {
    background: #003333 !important;
    border: 1px solid var(--neon-blue);
}
/* LABELS */
h3, label, .gr-checkbox label span {
    color: var(--neon-blue) !important;
    text-shadow: var(--neon-glow);
    font-weight: 600;
    text-transform: uppercase;
}
"""

BLOCKS_CSS_KWARGS = {"css": CYBERPUNK_CSS} if _GRADIO_MAJOR < 6 else {}
LAUNCH_CSS_KWARGS = {"css": CYBERPUNK_CSS} if _GRADIO_MAJOR >= 6 else {}


# --------------------------------------------------------------------------- #
# UI callbacks
# --------------------------------------------------------------------------- #

def on_lora_files_change(files):
    """Show one slider/label pair per uploaded LoRA file, hide the rest."""
    files = files or []
    updates = []
    for i in range(MAX_LORAS):
        if i < len(files):
            name = os.path.basename(files[i].name if hasattr(files[i], "name") else files[i])
            updates.append(gr.update(visible=True, label=f"Weight — {name}", value=1.0))
        else:
            updates.append(gr.update(visible=False))
    return updates


def on_check_compatibility(base_file, lora_files):
    if base_file is None:
        return "⚠️ Please load a base model first."
    if not lora_files:
        return "⚠️ Please load at least one LoRA file."

    base_path = base_file.name if hasattr(base_file, "name") else base_file
    lines = [f"Base model: {os.path.basename(base_path)}", ""]
    try:
        for lf in lora_files:
            lora_path = lf.name if hasattr(lf, "name") else lf
            report = check_compatibility(base_path, lora_path)
            total = len(report.matched) + len(report.unmatched) + len(report.shape_mismatches)
            lines.append(f"— {report.lora_file} —")
            lines.append(f"  matched:          {len(report.matched)}/{total}")
            lines.append(f"  unmatched:        {len(report.unmatched)}")
            lines.append(f"  shape mismatches: {len(report.shape_mismatches)}")
            if report.unmatched:
                sample = report.unmatched[:5]
                lines.append(f"  unmatched sample: {sample}")
            if report.shape_mismatches:
                sample = report.shape_mismatches[:5]
                lines.append(f"  shape mismatch sample: {sample}")
            if len(report.matched) == 0:
                lines.append("  ❌ No layers matched — this LoRA is likely NOT compatible with this base model.")
            elif report.unmatched or report.shape_mismatches:
                lines.append("  ⚠️ Partial match — merge will proceed but some layers will be skipped.")
            else:
                lines.append("  ✅ Full match — safe to merge.")
            lines.append("")
    except Exception as e:
        return "❌ Error checking compatibility:\n" + "".join(traceback.format_exception(e))

    return "\n".join(lines)


def on_merge(
    base_file,
    lora_files,
    w1, w2, w3, w4, w5, w6, w7, w8,
    compute_dtype_label,
    output_dtype_label,
    device_label,
    output_dir,
    output_name,
    progress=gr.Progress(track_tqdm=False),
):
    if base_file is None:
        return "⚠️ Please load a base model first.", None
    if not lora_files:
        return "⚠️ Please load at least one LoRA file.", None

    base_path = base_file.name if hasattr(base_file, "name") else base_file
    weights = [w1, w2, w3, w4, w5, w6, w7, w8][: len(lora_files)]

    device = "cuda" if (device_label == "cuda" and torch.cuda.is_available()) else "cpu"
    compute_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[compute_dtype_label]
    out_dtype = DTYPE_MAP[output_dtype_label]

    log_lines = [f"Loading base model on {device}..."]
    progress(0.05, desc="Loading base model")

    try:
        base_state = load_state_dict(base_path, device=device)
    except Exception as e:
        return "❌ Failed to load base model:\n" + "".join(traceback.format_exception(e)), None

    log_lines.append(f"Base model loaded: {len(base_state)} tensors.")

    total_matched, total_unmatched, total_shape = 0, 0, 0
    n = len(lora_files)
    for idx, (lf, w) in enumerate(zip(lora_files, weights)):
        lora_path = lf.name if hasattr(lf, "name") else lf
        progress((idx + 1) / (n + 1), desc=f"Merging {os.path.basename(lora_path)}")
        log_lines.append(f"\nMerging {os.path.basename(lora_path)} at weight {w}...")
        try:
            report = merge_lora_file(base_state, lora_path, weight=w, compute_dtype=compute_dtype)
        except Exception as e:
            return (
                "\n".join(log_lines) + "\n❌ Error during merge:\n" + "".join(traceback.format_exception(e)),
                None,
            )
        total_matched += len(report.matched)
        total_unmatched += len(report.unmatched)
        total_shape += len(report.shape_mismatches)
        log_lines.append(
            f"  matched {len(report.matched)}, unmatched {len(report.unmatched)}, "
            f"shape mismatches {len(report.shape_mismatches)}"
        )

    os.makedirs(output_dir, exist_ok=True)
    name = output_name.strip() or f"krea2_turbo_merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if not name.endswith(".safetensors"):
        name += ".safetensors"
    output_path = os.path.join(output_dir, name)

    progress(0.95, desc="Saving merged model")
    log_lines.append(f"\nSaving merged model to {output_path} ...")
    try:
        save_merged(base_state, output_path, out_dtype)
    except Exception as e:
        return "\n".join(log_lines) + "\n❌ Error saving file:\n" + "".join(traceback.format_exception(e)), None

    log_lines.append("Done.")
    log_lines.append(f"\nTotals — matched: {total_matched}, unmatched: {total_unmatched}, shape mismatches: {total_shape}")
    if total_unmatched or total_shape:
        log_lines.append("⚠️ Some LoRA layers were skipped. The merge still completed, but review the numbers above.")

    return "\n".join(log_lines), output_path


# --------------------------------------------------------------------------- #
# UI layout
# --------------------------------------------------------------------------- #

with gr.Blocks(title="Krea 2 Turbo LoRA Merger", **BLOCKS_CSS_KWARGS) as demo:
    gr.Markdown(
        "# Krea 2 Turbo LoRA Merger\n"
        "Permanently bake one or more LoRAs into a base checkpoint. "
        "Fully offline — nothing here calls the internet.",
        elem_id="main_title",
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Load files")
            base_file = gr.File(label="Base model (.safetensors)", file_types=[".safetensors"])
            lora_files = gr.File(
                label="LoRA file(s) (.safetensors)",
                file_types=[".safetensors"],
                file_count="multiple",
            )

            gr.Markdown("### 2. LoRA weights")
            weight_sliders = []
            for i in range(MAX_LORAS):
                s = gr.Slider(
                    minimum=-2.0, maximum=2.0, value=1.0, step=0.05,
                    label=f"Weight — LoRA {i + 1}", visible=False,
                )
                weight_sliders.append(s)

            lora_files.change(
                fn=on_lora_files_change,
                inputs=[lora_files],
                outputs=weight_sliders,
            )

            gr.Markdown("### 3. Settings")
            compute_dtype = gr.Radio(COMPUTE_DTYPES, value="fp32", label="Compute precision (during merge math)")
            output_dtype = gr.Radio(OUTPUT_DTYPES, value="keep original", label="Output precision")
            device_choice = gr.Radio(["cpu", "cuda"], value="cpu", label="Device (cuda requires a working GPU + CUDA torch build)")
            output_dir = gr.Textbox(label="Output folder", value="./output", placeholder="./output")
            output_name = gr.Textbox(label="Output filename (without extension optional)", placeholder="krea2_turbo_merged")

        with gr.Column(scale=1):
            gr.Markdown("### 4. Check & merge")
            check_btn = gr.Button("Check compatibility (dry run)")
            compat_output = gr.Textbox(label="Compatibility report", lines=14, interactive=False)

            merge_btn = gr.Button("Merge & Save", variant="primary")
            merge_log = gr.Textbox(label="Merge log", lines=16, interactive=False)
            merged_file = gr.File(label="Merged model output", interactive=False)

    check_btn.click(
        fn=on_check_compatibility,
        inputs=[base_file, lora_files],
        outputs=[compat_output],
    )

    merge_btn.click(
        fn=on_merge,
        inputs=[base_file, lora_files, *weight_sliders, compute_dtype, output_dtype, device_choice, output_dir, output_name],
        outputs=[merge_log, merged_file],
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True, **LAUNCH_CSS_KWARGS)
