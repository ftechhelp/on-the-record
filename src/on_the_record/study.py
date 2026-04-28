"""Generate study documents from completed transcripts using Gemini."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"
_GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def load_gemini_api_key() -> str | None:
    """Return the Gemini API key if configured."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    return key or None


def default_study_output_path(transcript_path: str | Path) -> Path:
    """Return the default Markdown study-document path for a transcript."""
    path = Path(transcript_path)
    return path.with_name(f"{path.stem}_study.md")


def generate_study_document(
    transcript_text: str,
    *,
    api_key: str,
    model: str = DEFAULT_GEMINI_MODEL,
    timeout: float = 120.0,
) -> str:
    """Ask Gemini to turn a transcript into a polished Markdown study document."""
    transcript_text = transcript_text.strip()
    if not transcript_text:
        raise ValueError("Cannot generate a study document from an empty transcript.")

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": _build_study_prompt(transcript_text),
                    }
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.35,
        },
    }
    request = urllib.request.Request(
        _build_gemini_url(model, api_key),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_data = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini request failed with HTTP {exc.code}: {details}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Gemini request failed: {exc.reason}") from exc

    document = _extract_text(json.loads(response_data)).strip()
    if not document:
        raise RuntimeError("Gemini returned an empty study document.")

    return _strip_markdown_fence(document)


def write_study_document(
    transcript_path: str | Path,
    output_path: str | Path,
    *,
    api_key: str,
    model: str = DEFAULT_GEMINI_MODEL,
) -> Path:
    """Generate and write a Markdown study document for *transcript_path*."""
    transcript_path = Path(transcript_path)
    output_path = Path(output_path)
    transcript_text = transcript_path.read_text(encoding="utf-8")
    document = generate_study_document(
        transcript_text,
        api_key=api_key,
        model=model,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document.rstrip() + "\n", encoding="utf-8")
    return output_path


def _build_gemini_url(model: str, api_key: str) -> str:
    model = urllib.parse.quote(model, safe="")
    key = urllib.parse.quote(api_key, safe="")
    return _GEMINI_ENDPOINT.format(model=model) + f"?key={key}"


def _build_study_prompt(transcript_text: str) -> str:
    return f"""You are an expert educator and technical writer.

Turn the transcript below into a useful Markdown study document. Infer the subject matter and choose a document style that fits the content, such as study notes, a meeting brief, a tutorial, a lecture guide, a troubleshooting guide, or an action-oriented reference.

Requirements:
- Return only Markdown, with no code fence wrapping the whole response.
- Give the document a clear title.
- Preserve important facts, decisions, examples, caveats, and terminology from the transcript.
- Organize the material for learning and review, not as a verbatim transcript.
- Include sections that fit the content, such as key takeaways, concepts, timeline, decisions, action items, glossary, questions to review, examples, or next steps.
- Use speaker labels only when they help explain context or decisions.
- If the transcript is short or informal, still produce a concise useful study note.

Transcript:
{transcript_text}
"""


def _extract_text(response: dict) -> str:
    parts: list[str] = []
    for candidate in response.get("candidates", []):
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            text = part.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts)


def _strip_markdown_fence(document: str) -> str:
    stripped = document.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped

