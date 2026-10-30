"""Calibration-set sampling for Brevitas PTQ, seeded per repeat.

Every ViT-branch manifest row is stochastic (unlike the LLM branch, there
is no deterministic exception): the calibration set is always sampled
randomly from data_prep's ImageNet-1K calibration pool
(calibration_protocol_vit.md Section 6), so every row gets NUM_REPEATS
independent draws. seed_for_repeat mirrors
quantization/llm_llamacpp/imatrix_gen.py's exactly -- a fixed global seed
would make every repeat draw the same images, defeating the point of
repeating at all.
"""

from __future__ import annotations

NUM_REPEATS = 3
BASE_SEED = 42


def seed_for_repeat(repeat_id: int) -> int:
    return BASE_SEED + (repeat_id - 1)


def load_calibration_batch(model, calib_set_size: int, repeat_id: int):
    """calib_set_size images sampled from data_prep's ImageNet-1K
    calibration pool (train split, never eval -- no data leakage), seeded
    per repeat_id so different repeats draw a genuinely different sample,
    preprocessed exactly as `model` (a timm model instance) expects, and
    stacked into a single batch tensor."""
    import torch
    from PIL import Image
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform

    from data_prep.imagenet import get_calibration_batch

    config = resolve_data_config({}, model=model)
    preprocess = create_transform(**config)

    samples = get_calibration_batch(
        repetition_idx=0, batch_size=calib_set_size, seed=seed_for_repeat(repeat_id)
    )
    tensors = [preprocess(Image.open(sample["path"]).convert("RGB")) for sample in samples]
    return torch.stack(tensors)
