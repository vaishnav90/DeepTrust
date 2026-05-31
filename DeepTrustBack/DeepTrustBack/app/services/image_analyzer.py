import base64
import os
import traceback

import requests
from dotenv import load_dotenv

load_dotenv()


class ImageAnalyzer:
    """
    Image analyzer:
    - Accepts image input as raw bytes OR a base64-encoded string/bytes.
    - Calls the SAME image inference endpoint used by `VideoAnalyzer` (HUGGINGFACE_IMAGE_API_URL).
    """

    def __init__(self):
        self.api_url = os.getenv("HUGGINGFACE_IMAGE_API_URL")
        self.api_key = os.getenv("HUGGINGFACE_API_KEY")

        # Keep behavior consistent with existing analyzers: fail fast if misconfigured.
        if not self.api_key:
            raise ValueError("HUGGINGFACE_API_KEY is not set")
        if not self.api_url:
            raise ValueError("HUGGINGFACE_IMAGE_API_URL is not set")

    @staticmethod
    def _coerce_image_bytes(image_input) -> bytes:
        """
        Accept:
        - raw bytes (jpg/png/webp/etc)
        - base64 string
        - base64-as-bytes (utf-8)
        """
        if image_input is None:
            return b""

        if isinstance(image_input, (bytes, bytearray)):
            b = bytes(image_input)
            # If it's a base64-encoded *string* serialized as bytes, decode it.
            try:
                s = b.decode("utf-8").strip()
            except Exception:
                return b
            try:
                decoded = base64.b64decode(s, validate=True)
                return decoded if decoded else b
            except Exception:
                return b

        if isinstance(image_input, str):
            s = image_input.strip()
            try:
                decoded = base64.b64decode(s, validate=True)
                return decoded if decoded else b""
            except Exception:
                return s.encode("utf-8", errors="ignore")

        return b""

    def _headers(self) -> dict:
        return {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def analyze_image(self, image_input, *, filename: str | None = None, content_type: str | None = None) -> dict:
        """
        Returns the raw inference response from the image endpoint (dict/list),
        or {"error": "..."} on failure.
        """
        try:
            if not self.api_url:
                return {"error": "HUGGINGFACE_IMAGE_API_URL is not set"}

            image_bytes = self._coerce_image_bytes(image_input)
            if not image_bytes:
                return {"error": "Empty image payload"}

            b64_image = base64.b64encode(image_bytes).decode("utf-8")
            payload = {"inputs": b64_image, "parameters": {}}

            timeout_s = int(os.getenv("HUGGINGFACE_TIMEOUT", "60"))
            response = requests.post(
                self.api_url,
                headers=self._headers(),
                json=payload,
                timeout=timeout_s,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            preview = None
            try:
                resp = getattr(e, "response", None)
                if resp is not None:
                    preview = (resp.text or "")[:2000]
            except Exception:
                preview = None
            return {"error": f"Image inference failed: {type(e).__name__}: {e}", "upstream_body_preview": preview}
        except Exception as e:
            return {"error": f"Unexpected error: {type(e).__name__}: {e}", "traceback": traceback.format_exc()[:4000]}

