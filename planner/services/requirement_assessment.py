"""Semantic completeness check for the planner's evidence requirements.

The checker judges only the supplied evidence. It never searches, invents
sources, or treats a retrieved candidate as proof merely because it exists.
Failures are returned as unavailable so the legacy deterministic sufficiency
path remains operational.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Mapping

from llm.router import router

logger = logging.getLogger(__name__)

ASSESSMENT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "evidence_requirement_assessment",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "requirements": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "status": {"type": "string", "enum": ["satisfied", "partial", "missing", "conflict"]},
                            "evidence_indices": {"type": "array", "items": {"type": "integer"}},
                            "reason": {"type": "string"},
                        },
                        "required": ["id", "status", "evidence_indices", "reason"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["requirements"],
            "additionalProperties": False,
        },
    },
}

SYSTEM_PROMPT = """You are QueryPilot's evidence completeness auditor. Decide whether each requirement is established by the supplied evidence, not whether it merely sounds related. Use only the provided evidence. Mark satisfied only when the evidence directly supports the complete requirement. Mark partial when it supports only part, missing when it does not establish it, and conflict when supplied sources materially disagree. Never infer missing facts. Cite evidence_indices using the zero-based indices supplied. If no evidence index supports a requirement, return an empty list. Keep reason concise."""


def _read(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def assess_requirements(
    *,
    requirements: list[dict[str, Any]],
    evidence: list[Any],
) -> dict[str, Any] | None:
    if not requirements:
        return None

    evidence_payload = []
    for index, item in enumerate(evidence[:16]):
        content = _read(item, "content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, default=str)
        metadata = _read(item, "metadata", {})
        if not isinstance(metadata, Mapping):
            metadata = {}
        evidence_payload.append({
            "index": index,
            "source_type": str(_read(item, "source_type", "unknown"))[:40],
            "document_id": _read(item, "document_id"),
            "document_name": str(_read(item, "document_name", "") or "")[:160],
            "table_names": metadata.get("table_names") or metadata.get("table_name") or [],
            "content": content[:1400],
        })

    payload = {
        "requirements": [
            {
                "id": item["id"],
                "kind": item["kind"],
                "type": item["type"],
                "description": item["description"],
                "depends_on": item["depends_on"],
            }
            for item in requirements
        ],
        "evidence": evidence_payload,
    }
    try:
        response = router.generate(
            task="planner",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)},
            ],
            temperature=0.0,
            response_format=ASSESSMENT_SCHEMA,
            include_reasoning=False,
        )
        content = response.choices[0].message.content
        result = json.loads(content)
        raw_items = result.get("requirements") if isinstance(result, dict) else None
        if not isinstance(raw_items, list):
            raise ValueError("assessment requirements must be a list")

        requested = {item["id"] for item in requirements}
        validated = []
        seen = set()
        valid_indices = set(range(len(evidence_payload)))
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            requirement_id = str(item.get("id") or "")
            status = str(item.get("status") or "missing").lower()
            if requirement_id not in requested or requirement_id in seen:
                continue
            seen.add(requirement_id)
            if status not in {"satisfied", "partial", "missing", "conflict"}:
                status = "missing"
            indices = []
            for value in item.get("evidence_indices", []) or []:
                if isinstance(value, int) and value in valid_indices and value not in indices:
                    indices.append(value)
            if status == "satisfied" and not indices:
                status = "missing"
            validated.append({
                "id": requirement_id,
                "status": status,
                "evidence_indices": indices,
                "reason": str(item.get("reason") or "")[:400],
            })
        for missing_id in requested - seen:
            validated.append({
                "id": missing_id,
                "status": "missing",
                "evidence_indices": [],
                "reason": "The auditor did not return an assessment for this requirement.",
            })
        complete = bool(validated) and all(item["status"] == "satisfied" for item in validated)
        return {"complete": complete, "requirements": validated}
    except Exception:
        logger.exception("Evidence requirement assessment unavailable")
        return None
