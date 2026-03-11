"""Model registry and configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from model_eval.models.base import ModelAdapter

# Maps model name -> (module_path, class_name)
# Lazy imports to avoid loading heavy dependencies at startup.
AESTHETIC_MODELS: dict[str, tuple[str, str]] = {
    "siglip-v2.5": (
        "model_eval.models.aesthetic.siglip_v25",
        "SigLIPAestheticAdapter",
    ),
    "clip-mlp": (
        "model_eval.models.aesthetic.clip_mlp",
        "CLIPMLPAestheticAdapter",
    ),
    "q-align": (
        "model_eval.models.aesthetic.q_align",
        "QAlignAdapter",
    ),
    "pyiqa-topiq": (
        "model_eval.models.aesthetic.pyiqa_suite",
        "PyIQAAdapter",
    ),
    "pyiqa-clipiqa": (
        "model_eval.models.aesthetic.pyiqa_suite",
        "PyIQACLIPIQAAdapter",
    ),
    "pyiqa-maniqa": (
        "model_eval.models.aesthetic.pyiqa_suite",
        "PyIQAMANIQAAdapter",
    ),
    "pyiqa-nima": (
        "model_eval.models.aesthetic.pyiqa_suite",
        "PyIQANIMAAdapter",
    ),
    "pyiqa-musiq": (
        "model_eval.models.aesthetic.pyiqa_suite",
        "PyIQAMUSIQAdapter",
    ),
    "unipercept": (
        "model_eval.models.aesthetic.unipercept",
        "UniPerceptAdapter",
    ),
    "artimuse": (
        "model_eval.models.aesthetic.artimuse",
        "ArtiMuseAdapter",
    ),
}

CAPTION_MODELS: dict[str, tuple[str, str]] = {
    "blip2": (
        "model_eval.models.caption.blip2",
        "BLIP2Adapter",
    ),
    "qwen3-vl": (
        "model_eval.models.caption.qwen3_vl",
        "Qwen3VLAdapter",
    ),
    "qwen3-vl-int4": (
        "model_eval.models.caption.qwen3_vl_quant",
        "Qwen3VLQuantAdapter",
    ),
    "qwen3-vl-2b": (
        "model_eval.models.caption.qwen3_vl_quant",
        "Qwen3VL2BAdapter",
    ),
    "qwen3-vl-4b": (
        "model_eval.models.caption.qwen3_vl_quant",
        "Qwen3VL4BAdapter",
    ),
    "qwen3-vl-concise": (
        "model_eval.models.caption.qwen3_vl_quant",
        "Qwen3VLConciseAdapter",
    ),
    "qwen3-vl-natural": (
        "model_eval.models.caption.qwen3_vl_quant",
        "Qwen3VLNaturalAdapter",
    ),
    "qwen3-vl-observer": (
        "model_eval.models.caption.qwen3_vl_quant",
        "Qwen3VLObserverAdapter",
    ),
    "gemma3": (
        "model_eval.models.caption.gemma3",
        "Gemma3Adapter",
    ),
    "molmo": (
        "model_eval.models.caption.molmo",
        "MolmoAdapter",
    ),
    "florence2": (
        "model_eval.models.caption.florence2",
        "Florence2Adapter",
    ),
    "joycaption": (
        "model_eval.models.caption.joycaption",
        "JoyCaptionAdapter",
    ),
    "minicpm-v": (
        "model_eval.models.caption.minicpm_v",
        "MiniCPMVAdapter",
    ),
    "internvl": (
        "model_eval.models.caption.internvl",
        "InternVLAdapter",
    ),
    "gpt-4o": (
        "model_eval.models.caption.gpt4o",
        "GPT4oAdapter",
    ),
    "copilot": (
        "model_eval.models.caption.copilot",
        "CopilotAdapter",
    ),
    "qwen3.5-vl-2b": (
        "model_eval.models.caption.ollama_generic",
        "Qwen35VL2BAdapter",
    ),
    "qwen3.5-vl-4b": (
        "model_eval.models.caption.ollama_generic",
        "Qwen35VL4BAdapter",
    ),
    "qwen3.5-vl-9b": (
        "model_eval.models.caption.ollama_generic",
        "Qwen35VL9BAdapter",
    ),
    "qwen2.5-vl-3b": (
        "model_eval.models.caption.ollama_generic",
        "Qwen25VL3BAdapter",
    ),
    "qwen2.5-vl-7b": (
        "model_eval.models.caption.ollama_generic",
        "Qwen25VL7BAdapter",
    ),
}

ALL_MODELS: dict[str, tuple[str, str]] = {**AESTHETIC_MODELS, **CAPTION_MODELS}


def get_adapter(name: str) -> ModelAdapter:
    """Instantiate a model adapter by name (lazy import)."""
    import importlib

    if name not in ALL_MODELS:
        raise ValueError(
            f"Unknown model {name!r}. Available: {sorted(ALL_MODELS.keys())}"
        )
    module_path, class_name = ALL_MODELS[name]
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls()


def list_model_names(category: str | None = None) -> list[str]:
    """Return available model names, optionally filtered by category."""
    if category == "aesthetic":
        return sorted(AESTHETIC_MODELS.keys())
    elif category == "caption":
        return sorted(CAPTION_MODELS.keys())
    return sorted(ALL_MODELS.keys())
