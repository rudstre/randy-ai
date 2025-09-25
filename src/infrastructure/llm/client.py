"""
Vertex AI REST client for LLM interactions.
"""
import json
import logging
from typing import Optional, Dict, Any, List

import requests
import google.auth
import google.auth.transport.requests
from google.oauth2 import service_account

from ...config import VERTEX_LOCATION, MODEL_NAME, LLM_TIMEOUT, MAX_OUTPUT_TOKENS

logger = logging.getLogger("llm_client")

class VertexRestClient:
    """REST-based client for Vertex AI Gemini models."""
    
    def __init__(self,
                 project: str,
                 location: str = VERTEX_LOCATION,
                 model: str = MODEL_NAME,
                 credentials_json: Optional[str] = None,
                 timeout: int = LLM_TIMEOUT):
        self.project = project
        self.location = location
        self.model = model
        self.credentials_json = credentials_json
        self.base_url = f"https://{self.location}-aiplatform.googleapis.com/v1"
        self.model_resource = f"projects/{self.project}/locations/{self.location}/publishers/google/models/{self.model}"
        self._token = None
        self.timeout = timeout

    def _refresh_token(self):
        """Refresh the OAuth token for API calls."""
        if self.credentials_json:
            creds = service_account.Credentials.from_service_account_file(
                self.credentials_json,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
        else:
            creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        
        auth_req = google.auth.transport.requests.Request()
        creds.refresh(auth_req)
        self._token = creds.token

    def _ensure_token(self):
        """Ensure we have a valid token, refreshing if needed."""
        if not self._token:
            self._refresh_token()

    def generate_content(
        self,
        prompt_text: str,
        temperature: float = 0.0,
        max_output_tokens: int = MAX_OUTPUT_TOKENS,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """Generate content using the Vertex AI REST API."""
        self._ensure_token()
        url = f"{self.base_url}/{self.model_resource}:generateContent"

        body: Dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt_text}],
                }
            ],
            "generationConfig": {
                "temperature": float(temperature),
                "maxOutputTokens": int(max_output_tokens),
            },
        }

        if top_k is not None:
            body["generationConfig"]["topK"] = int(top_k)
        if top_p is not None:
            body["generationConfig"]["topP"] = float(top_p)
        if stop_sequences:
            body["generationConfig"]["stopSequences"] = list(stop_sequences)

        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }

        resp = requests.post(url, headers=headers, json=body, timeout=self.timeout)
        if resp.status_code >= 400:
            raise RuntimeError(f"Vertex REST error {resp.status_code}: {resp.text}")

        return self._parse_response_text(resp.json())

    def _parse_response_text(self, resp_json: Dict[str, Any]) -> str:
        """
        Parse response JSON to extract text content.
        Tries Vertex schema first, then falls back to alternatives.
        """
        try:
            # Vertex schema: candidates[0].content.parts[0].text
            cands = resp_json.get("candidates", [])
            if cands:
                first = cands[0]
                content = first.get("content", {})
                parts = content.get("parts", [])
                if parts and isinstance(parts, list):
                    # Find first part with "text"
                    for p in parts:
                        if isinstance(p, dict) and "text" in p and isinstance(p["text"], str):
                            return p["text"]
                # Some responses put text directly in content
                if isinstance(content, dict) and isinstance(content.get("text"), str):
                    return content["text"]
            
            # Direct text fallback
            if isinstance(resp_json.get("text"), str):
                return resp_json["text"]
                
        except Exception:
            pass
        
        # Last resort: return JSON for inspection
        return json.dumps(resp_json, separators=(",", ":"))

    def generate_json(self, prompt: str) -> Dict[str, Any]:
        """
        Generate JSON response from LLM with defensive parsing.
        Automatically appends instruction to respond with JSON only.
        """
        prompt_json = prompt.strip() + "\n\nRespond ONLY with minified JSON."
        logger.debug("Sending JSON prompt to LLM...")
        
        try:
            text = self.generate_content(prompt_json, temperature=0.0, max_output_tokens=MAX_OUTPUT_TOKENS)
        except Exception as e:
            logger.error("LLM request failed: %s", e)
            raise

        logger.debug("Raw LLM output: %s", repr(text))

        # Try to parse JSON
        try:
            parsed = json.loads(text)
            logger.debug("Parsed JSON successfully")
            return parsed
        except Exception as e:
            logger.warning("json.loads failed: %s", e)
            # Try extracting JSON from text
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    parsed = json.loads(text[start:end+1])
                    logger.debug("Parsed JSON from substring successfully")
                    return parsed
                except Exception as e2:
                    logger.warning("Substring parse also failed: %s", e2)
            
            raise ValueError(f"LLM did not return valid JSON: {text}")
