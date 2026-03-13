"""UniPercept perceptual scoring — IAA / IQA / ISTA.

Wraps the ``unipercept-reward`` package (CVPR 2026, InternVL2.5-8B fine-tuned)
loaded with 4-bit NF4 quantization to fit in 16 GB VRAM.

Scores are produced on a 0-100 scale by the model and normalized to 0-10 for
consistency with the rest of imganalyzer (e.g. ``aesthetic_score``).

Three dimensions:
  * **IAA** — Image Aesthetic Appeal
  * **IQA** — Image Quality Assessment
  * **ISTA** — Image Structure & Texture Analysis
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def _get_hf_token() -> str | None:
    """Return a HuggingFace token from the environment or cached login."""
    import os

    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import get_token
        return get_token()
    except Exception:
        return None


# ── Label mapping (0-10 scale) ────────────────────────────────────────────────

_LABEL_THRESHOLDS: list[tuple[float, str]] = [
    (9.0, "Masterpiece"),
    (8.0, "Excellent"),
    (7.0, "Very Good"),
    (6.0, "Good"),
    (5.0, "Average"),
    (3.0, "Below Average"),
    (0.0, "Poor"),
]


def score_to_label(score: float) -> str:
    """Map a 0-10 score to a human-readable label."""
    for threshold, label in _LABEL_THRESHOLDS:
        if score >= threshold:
            return label
    return "Poor"


# ── Model holder (singleton per process) ──────────────────────────────────────

class _ModelHolder:
    """Lazy-loaded, reusable model context."""

    def __init__(self) -> None:
        self.model: Any = None
        self.tokenizer: Any = None
        self.gen_cfg: dict[str, Any] = {}
        self.transform: Any = None
        self.device: str = "cuda"
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    # Local cache for quantized weights (avoids re-quantising on every start).
    _QUANT_CACHE = Path.home() / ".cache" / "imganalyzer" / "unipercept_4bit"

    def _apply_monkey_patches(self) -> None:
        """Apply transformers 5.0 compat patches (idempotent)."""
        import torch
        from unipercept_reward.internvl.model.internvl_chat.modeling_unipercept import (
            InternVLChatModel,
        )
        from unipercept_reward.internvl.model.internvl_chat.configuration_internvl_chat import (
            InternVLChatConfig,
        )
        from unipercept_reward.internvl.model.internvl_chat import modeling_intern_vit

        # 1. Config diffing calls __init__() with no args; default llm_config
        #    has architectures=[''] which raises ValueError.
        InternVLChatConfig.has_no_defaults_at_init = True

        # 2. InternVisionEncoder.__init__ calls torch.linspace().item()
        #    which fails on meta device used by device_map="auto".
        _orig_linspace = torch.linspace

        def _cpu_linspace(*args: Any, **kwargs: Any) -> Any:
            kwargs.setdefault("device", "cpu")
            return _orig_linspace(*args, **kwargs)

        modeling_intern_vit.torch.linspace = _cpu_linspace

        # 3. BnB quantizer accesses model.all_tied_weights_keys.
        if not hasattr(InternVLChatModel, "all_tied_weights_keys"):
            InternVLChatModel.all_tied_weights_keys = {}

    def load(self) -> None:
        if self._loaded:
            return

        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(
                "UniPercept requires CUDA, but no CUDA device is available."
            )
        from transformers import AutoTokenizer, BitsAndBytesConfig
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        from unipercept_reward.internvl.model.internvl_chat.modeling_unipercept import (
            InternVLChatModel,
        )

        self._apply_monkey_patches()

        hf_token = _get_hf_token()
        cache_dir = self._QUANT_CACHE
        use_cache = cache_dir.exists() and (cache_dir / "config.json").exists()

        if use_cache:
            # ── Fast path: load pre-quantized weights from local cache ──
            log.info("Loading cached quantized UniPercept from %s …", cache_dir)
            self.model = InternVLChatModel.from_pretrained(
                str(cache_dir),
                torch_dtype=torch.bfloat16,
                use_flash_attn=False,
                trust_remote_code=True,
                device_map="auto",
                max_memory={0: "14GiB", "cpu": "24GiB"},
            ).eval()

            self.tokenizer = AutoTokenizer.from_pretrained(
                str(cache_dir), trust_remote_code=True, use_fast=False,
            )
        else:
            # ── First-time: quantize from HuggingFace Hub, then cache ──
            model_path = "Thunderbolt215215/UniPercept"
            log.info(
                "First-time load: quantizing UniPercept from %s "
                "(this may take several minutes) …",
                model_path,
            )

            if hf_token is None:
                log.warning(
                    "No HuggingFace token found. UniPercept is a gated model — "
                    "run `huggingface-cli login` to authenticate."
                )

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )

            try:
                self.model = InternVLChatModel.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    use_flash_attn=False,
                    trust_remote_code=True,
                    quantization_config=quant_config,
                    device_map="auto",
                    max_memory={0: "14GiB", "cpu": "24GiB"},
                    token=hf_token,
                ).eval()

                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path, trust_remote_code=True, use_fast=False,
                    token=hf_token,
                )
            except OSError as exc:
                if "gated repo" in str(exc) or "401" in str(exc):
                    raise OSError(
                        "Cannot access gated model Thunderbolt215215/UniPercept. "
                        "1) Run `huggingface-cli login` with a valid token. "
                        "2) Accept the model license at "
                        "https://huggingface.co/Thunderbolt215215/UniPercept"
                    ) from exc
                raise

            # Persist quantized weights so future loads are fast.
            log.info("Caching quantized model to %s …", cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(str(cache_dir))
            self.tokenizer.save_pretrained(str(cache_dir))
            log.info("Quantized model cached — subsequent loads will be fast")

        self.gen_cfg = {"max_new_tokens": 512, "do_sample": False}

        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        self._loaded = True
        log.info("UniPercept model loaded")

    def unload(self) -> None:
        if not self._loaded:
            return
        import torch

        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        self.transform = None
        self._loaded = False

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
        log.info("UniPercept model unloaded")


# Module-level singleton
_holder = _ModelHolder()


def load_model() -> None:
    """Pre-load the model (call before processing a batch)."""
    _holder.load()


def unload_model() -> None:
    """Release GPU memory (call after batch completes)."""
    _holder.unload()


# ── Public API ────────────────────────────────────────────────────────────────

_TASKS: dict[str, str] = {
    "iaa": "aesthetics",
    "iqa": "quality",
    "ista": "structure and texture richness",
}


def analyze(image_path: Path) -> dict[str, Any]:
    """Run UniPercept perceptual scoring on a single image.

    Returns a dict with keys:
      perception_iaa, perception_iaa_label,
      perception_iqa, perception_iqa_label,
      perception_ista, perception_ista_label
    All scores are on a 0-10 scale.
    """
    import torch
    from PIL import Image
    from imganalyzer.readers import open_as_pil

    _holder.load()

    img = open_as_pil(image_path)
    # Resize large images to avoid excessive memory usage
    MAX_DIM = 1280
    if max(img.size) > MAX_DIM:
        ratio = MAX_DIM / max(img.size)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    pixel_values = _holder.transform(img).unsqueeze(0)
    pixel_values = pixel_values.to(_holder.model.device, dtype=torch.bfloat16)

    result: dict[str, Any] = {}

    with torch.inference_mode():
        for key, desc in _TASKS.items():
            raw = _holder.model.score(
                _holder.device,
                _holder.tokenizer,
                pixel_values,
                _holder.gen_cfg,
                desc,
            )
            val = raw
            if isinstance(val, (list, tuple)):
                val = val[0]
            if hasattr(val, "item"):
                val = val.item()

            # Normalize from 0-100 → 0-10
            score_10 = round(float(val) / 10.0, 2)
            score_10 = max(0.0, min(10.0, score_10))

            result[f"perception_{key}"] = score_10
            result[f"perception_{key}_label"] = score_to_label(score_10)

    return result
