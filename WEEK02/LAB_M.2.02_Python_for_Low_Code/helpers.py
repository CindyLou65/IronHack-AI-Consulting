#Helpers Code for product generator notebook. These functions can be imported and reused across different notebooks or projects to avoid code duplication and improve maintainability.

# helpers.py

# To access in any notebook, simply import:
#from helpers import pil_to_base64_jpeg, parse_api_response_text

#Important: Restart Kernel Once
#When you first create helpers.py:
#Restart kernel
# Then import it
# Otherwise Jupyter might not detect it.

#What NOT to Put in helpers.py
# Don’t put:
# Dataset loading
# Notebook-specific variables
# products_df
# API keys
# Batch loops
# Helpers should only contain:
# Pure reusable functions
# No notebook state
# No global variables




import base64
import json
from io import BytesIO
from PIL import Image


# -------- Custom Exceptions --------

class PipelineError(Exception):
    """Base error for reusable pipeline utilities."""
    pass


class ResponseParseError(PipelineError):
    pass


class ImageEncodeError(PipelineError):
    pass


# -------- Image Encoding --------

def pil_to_base64_jpeg(pil_img: Image.Image) -> str:
    """
    Convert PIL image to base64 JPEG string.
    """
    try:
        if pil_img is None:
            raise ValueError("Received None image.")
        buf = BytesIO()
        pil_img.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        raise ImageEncodeError("Failed to encode PIL image.") from e


# -------- OpenAI Response Parsing --------

def _strip_markdown_fences(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            inner = parts[1].strip()
            if inner.lower().startswith("json"):
                inner = inner[4:].strip()
            return inner
    return text


def parse_api_response_text(raw_text: str) -> dict:
    """
    Parse model output text into JSON dictionary.
    """
    try:
        cleaned = _strip_markdown_fences(raw_text)
        return json.loads(cleaned)
    except Exception as e:
        preview = (raw_text or "")[:300].replace("\n", "\\n")
        raise ResponseParseError(
            f"Model output not valid JSON. Preview: {preview}"
        ) from e


