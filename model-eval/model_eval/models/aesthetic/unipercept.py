"""UniPercept unified perceptual assessment adapter.

Uses the ``unipercept-reward`` PyPI package which provides IAA (aesthetics),
IQA (quality), and ISTA (structure/texture) scores on a 0-100 scale.

The underlying model is InternVL2.5-8B fine-tuned for perceptual scoring.
We load it with 4-bit quantization to fit in 16 GB VRAM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from model_eval.models.base import ModelAdapter


class UniPerceptAdapter(ModelAdapter):
    name = "unipercept"
    category = "aesthetic"
    model_id = "Thunderbolt215215/UniPercept"

    def __init__(self) -> None:
        self._inferencer: Any = None

    def load(self, device: str = "cuda") -> None:
        import torch
        from unipercept_reward.inferencer import UniPerceptRewardInferencer

        # Monkey-patch the __init__ to add 4-bit quantization.
        # The package loads with .to(device) which requires full VRAM;
        # we intercept and use device_map + BitsAndBytes instead.
        self._inferencer = self._load_quantized(device)

    @staticmethod
    def _load_quantized(device: str) -> Any:
        """Load UniPercept with 4-bit quantization via bitsandbytes."""
        import torch
        from transformers import AutoTokenizer, BitsAndBytesConfig
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        # Import the model class from the unipercept_reward package
        from unipercept_reward.internvl.model.internvl_chat.modeling_unipercept import (
            InternVLChatModel,
        )
        from unipercept_reward.internvl.model.internvl_chat.configuration_internvl_chat import (
            InternVLChatConfig,
        )

        # Patch: transformers 5.0 calls __init__() with no args for config
        # diffing, but the default llm_config={'architectures': ['']} causes
        # ValueError. Mark the config so transformers skips the diff.
        InternVLChatConfig.has_no_defaults_at_init = True

        # Patch: InternVisionEncoder uses torch.linspace(...).item() during
        # __init__, which fails on the meta device used by device_map="auto".
        # Force linspace to use CPU.
        from unipercept_reward.internvl.model.internvl_chat import modeling_intern_vit
        _orig_linspace = torch.linspace

        def _cpu_linspace(*args, **kwargs):
            kwargs.setdefault("device", "cpu")
            return _orig_linspace(*args, **kwargs)

        modeling_intern_vit.torch.linspace = _cpu_linspace

        # Patch: transformers 5.0's BnB quantizer accesses
        # model.all_tied_weights_keys which doesn't exist on this
        # older InternVL model class.
        if not hasattr(InternVLChatModel, "all_tied_weights_keys"):
            InternVLChatModel.all_tied_weights_keys = {}

        model_path = "Thunderbolt215215/UniPercept"

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

        model = InternVLChatModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=False,
            trust_remote_code=True,
            quantization_config=quant_config,
            device_map="auto",
            max_memory={0: "14GiB", "cpu": "24GiB"},
        ).eval()

        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
        )

        gen_cfg = dict(max_new_tokens=512, do_sample=False)

        # Build the standard InternVL image transform
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        # Return a simple namespace-like object
        class _Ctx:
            pass
        ctx = _Ctx()
        ctx.model = model
        ctx.tokenizer = tokenizer
        ctx.gen_cfg = gen_cfg
        ctx.transform = transform
        ctx.device = device
        return ctx

    def run(self, image_path: Path) -> dict[str, Any]:
        import torch

        image = self.load_image(image_path)

        # Preprocess: apply InternVL transform, add batch dim
        pixel_values = self._inferencer.transform(image).unsqueeze(0)
        pixel_values = pixel_values.to(
            self._inferencer.model.device, dtype=torch.bfloat16
        )

        # Score all three dimensions
        tasks = {
            "iaa": "aesthetics",
            "iqa": "quality",
            "ista": "structure and texture richness",
        }

        scores: dict[str, float] = {}
        with torch.inference_mode():
            for key, desc in tasks.items():
                raw = self._inferencer.model.score(
                    self._inferencer.device,
                    self._inferencer.tokenizer,
                    pixel_values,
                    self._inferencer.gen_cfg,
                    desc,
                )
                val = raw
                if isinstance(val, (list, tuple)):
                    val = val[0]
                if hasattr(val, "item"):
                    val = val.item()
                scores[key] = round(float(val), 4)

        return {
            "score": scores["iaa"],
            "scale": "0-100",
            "details": scores,
        }

    def unload(self) -> None:
        if self._inferencer is not None:
            del self._inferencer.model
            del self._inferencer.tokenizer
            self._inferencer = None
        super().unload()
