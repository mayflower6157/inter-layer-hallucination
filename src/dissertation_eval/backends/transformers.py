from __future__ import annotations

import time
from typing import Any

from ..schema import BackendOutput, DecodingConfig, DecodingStrategy
from .base import GenerationBackend


def infer_finish_reason(
    token_ids: list[int], max_new_tokens: int, eos_token_id: int | list[int] | None
) -> str:
    """Classify whether generation stopped naturally or at the token limit."""

    if eos_token_id is None:
        eos_ids: set[int] = set()
    elif isinstance(eos_token_id, int):
        eos_ids = {eos_token_id}
    else:
        eos_ids = set(eos_token_id)
    if token_ids and token_ids[-1] in eos_ids:
        return "stop"
    if len(token_ids) >= max_new_tokens:
        return "length"
    return "stop"


class TransformersBackend(GenerationBackend):
    """Optional Hugging Face backend for text-only and vision-language baselines."""

    def __init__(self, model_id: str, revision: str, options: dict[str, Any]) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install the Hugging Face dependencies with: pip install -e '.[hf]'") from exc
        self.torch = torch
        self.model_id = model_id
        self.processor = None
        enable_thinking = options.get("enable_thinking")
        if enable_thinking is not None and not isinstance(enable_thinking, bool):
            raise ValueError("backend_options.enable_thinking must be true, false, or omitted")
        self.chat_template_options = (
            {"enable_thinking": enable_thinking} if enable_thinking is not None else {}
        )
        common = {"revision": revision, "trust_remote_code": bool(options.get("trust_remote_code", False))}
        if options.get("multimodal", False):
            from transformers import AutoModelForImageTextToText
            self.processor = AutoProcessor.from_pretrained(model_id, **common)
            self.tokenizer = self.processor.tokenizer
            model_cls = AutoModelForImageTextToText
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, **common)
            model_cls = AutoModelForCausalLM
        dtype_name = str(options.get("dtype", "bfloat16"))
        dtype = getattr(torch, dtype_name)
        self.model = model_cls.from_pretrained(model_id, torch_dtype=dtype,
                                               device_map=options.get("device_map", "auto"), **common)
        self.model.eval()

    def _inputs(self, prompt: str, image_path: str | None) -> dict[str, Any]:
        if self.processor is None:
            messages = [{"role": "user", "content": prompt}]
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **self.chat_template_options,
            )
            return self.tokenizer(rendered, return_tensors="pt").to(self.model.device)
        if image_path is None:
            raise ValueError("The multimodal backend requires image_path")
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                                     {"type": "text", "text": prompt}]}]
        rendered = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **self.chat_template_options,
        )
        return self.processor(text=[rendered], images=[image], padding=True, return_tensors="pt").to(self.model.device)

    def generate(self, prompt: str, image_path: str | None, config: DecodingConfig, seed: int) -> BackendOutput:
        torch = self.torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.cuda.reset_peak_memory_stats()
        inputs = self._inputs(prompt, image_path)
        kwargs: dict[str, Any] = {"max_new_tokens": config.max_new_tokens}
        if config.strategy == DecodingStrategy.SAMPLE:
            kwargs.update(do_sample=True, temperature=config.temperature, top_p=config.top_p)
        elif config.strategy == DecodingStrategy.BEAM:
            kwargs.update(do_sample=False, num_beams=config.num_beams)
        else:
            kwargs.update(do_sample=False, num_beams=1)
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(**inputs, **kwargs)
        elapsed_ms = (time.perf_counter() - started) * 1000
        prompt_length = int(inputs["input_ids"].shape[-1])
        generated = output[0, prompt_length:].tolist()
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        eos_token_id = self.model.generation_config.eos_token_id
        finish_reason = infer_finish_reason(generated, config.max_new_tokens, eos_token_id)
        return BackendOutput(text=text, finish_reason=finish_reason, prompt_tokens=prompt_length,
                             generated_tokens=len(generated), token_ids=generated,
                             metrics={"model_generate_ms": elapsed_ms})

    def peak_memory_bytes(self) -> int | None:
        if self.torch.cuda.is_available():
            return int(self.torch.cuda.max_memory_allocated())
        return None
