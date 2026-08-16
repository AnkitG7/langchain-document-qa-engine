"""Vision Model Provider and Image Understanding Subsystem.

Demonstrates:
- Native Multimodal Vision processing with Ollama (gemma4:cloud)
- Fallback abstractions for OpenAI GPT-4o-mini vision
- Factual visual description generation (chart types, axes, units, values, trends)
- Hallucination suppression prompt engineering for visual document elements
"""

import io
import os
import base64
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
from PIL import Image

import ollama
from config import settings

logger = logging.getLogger("DocMind.Vision")

VISION_EXTRACTION_SYSTEM_PROMPT = """You are an expert AI document visual analyst.
Analyze the provided visual element (chart, graph, diagram, table graphic, or picture) extracted from a document.

Provide a concise, factual summary covering:
1. **Visual Type:** (e.g. Bar Chart, Line Graph, Revenue Summary, Architecture Diagram, Flowchart, Photo)
2. **Title / Header:** (Exact title if visible, or infer concise subject)
3. **Axes, Units & Timeframes:** (e.g. USD in millions, quarters Q1-Q4 2024, volume in units)
4. **Key Numerical Values & Data Points:** (List exact numbers, percentages, bar heights, or coordinates visible)
5. **Main Trends & Relationships:** (Key comparisons, growth/decline, highlights)

Rules:
- Be strictly factual. Do NOT guess or hallucinate numbers not visible in the image.
- If text/labels in the image are blurry or absent, explicitly state that they are not visible.
"""


class VisionModelProvider:
    """Interfaces with multimodal vision models to generate structured text descriptions of visual elements."""

    def __init__(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.provider = (provider or os.getenv("VISION_PROVIDER", "ollama")).lower()
        self.model_name = model_name or os.getenv("VISION_MODEL_NAME", "gemma4:cloud")
        self.base_url = base_url or settings.ollama_base_url
        self._is_available = True

    def describe_image(
        self,
        image_input: Union[str, Path, bytes, Image.Image],
        context_hint: Optional[str] = None,
    ) -> str:
        """Analyzes an image and returns a factual textual description for RAG indexing."""
        # Convert image input to raw bytes
        image_bytes = self._to_image_bytes(image_input)
        if not image_bytes or len(image_bytes) < 100:
            return "[Empty or unreadable image]"

        prompt = VISION_EXTRACTION_SYSTEM_PROMPT
        if context_hint:
            prompt += f"\nContext Hint from Document: {context_hint}"

        if self.provider == "ollama":
            return self._describe_with_ollama(image_bytes, prompt)
        elif self.provider == "openai":
            return self._describe_with_openai(image_bytes, prompt)
        else:
            return self._describe_with_ollama(image_bytes, prompt)

    def _to_image_bytes(self, image_input: Union[str, Path, bytes, Image.Image]) -> Optional[bytes]:
        """Converts various image formats into raw PNG/JPEG bytes."""
        try:
            if isinstance(image_input, bytes):
                return image_input
            elif isinstance(image_input, (str, Path)):
                path_obj = Path(image_input)
                if not path_obj.exists():
                    return None
                with open(path_obj, "rb") as f:
                    return f.read()
            elif isinstance(image_input, Image.Image):
                buf = io.BytesIO()
                image_input.save(buf, format="PNG")
                return buf.getvalue()
        except Exception as e:
            logger.warning(f"Failed to convert image input to bytes: {e}")
            return None

    def _describe_with_ollama(self, image_bytes: bytes, prompt: str) -> str:
        """Invokes Ollama multimodal vision model."""
        try:
            client = ollama.Client(host=self.base_url)
            response = client.chat(
                model=self.model_name,
                messages=[{
                    "role": "user",
                    "content": prompt,
                    "images": [image_bytes],
                }],
            )
            content = response.get("message", {}).get("content", "").strip()
            return content or "[Visual description unavailable]"
        except Exception as e:
            logger.warning(f"Ollama vision processing failed: {e}")
            return f"[Visual element present; automated description unavailable: {str(e)[:100]}]"

    def _describe_with_openai(self, image_bytes: bytes, prompt: str) -> str:
        """Invokes OpenAI GPT-4o-mini vision model if configured."""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            b64_img = base64.b64encode(image_bytes).decode("utf-8")
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}},
                        ],
                    }
                ],
                max_tokens=500,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"OpenAI vision processing failed: {e}")
            return f"[Visual element present; description error: {str(e)[:100]}]"
