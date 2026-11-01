"""Validation and ordering helpers for planner evidence requirements.

This module is deliberately pure: it does not access Django, the vector store,
or the LLM provider. Invalid planner output is rejected safely and callers can
fall back to raw-question retrieval.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Mapping

MAX_REQUIREMENTS = 8
MAX_QUERIES_PER_REQUIREMENT = 3
MAX_QUERY_CHARS = 500
MAX_DEPENDENCY_DEPTH = 3
_ALLOWED_KINDS = {"text", "table", "either"}
_ALLOWED_TYPES = {
    "rule", "fact", "numeric", "entity", "status", "date", "comparison", "other"
}


def normalize_requirements(raw: Any) -> list[dict[str, Any]]:
    """Normalize planner requirements, validate references, and reject cycles."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("requirements must be a list")
    if len(raw) > MAX_REQUIREMENTS:
        raise ValueError(f"at most {MAX_REQUIREMENTS} requirements are allowed")

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, Mapping):
            raise ValueError(f"requirement {index} must be an object")
        requirement_id = str(item.get("id") or f"R{index}").strip()
        if not requirement_id or requirement_id in seen:
            raise ValueError("requirement IDs must be non-empty and unique")
        seen.add(requirement_id)

        kind = str(item.get("kind") or "text").strip().lower()
        if kind not in _ALLOWED_KINDS:
            raise ValueError(f"unsupported requirement kind: {kind}")
        requirement_type = str(item.get("type") or "other").strip().lower()
        if requirement_type not in _ALLOWED_TYPES:
            requirement_type = "other"

        description = str(item.get("description") or "").strip()[:1000]
        if not description:
            raise ValueError(f"requirement {requirement_id} has no description")

        queries: list[str] = []
        for query in item.get("queries", []) or []:
            value = str(query or "").strip()[:MAX_QUERY_CHARS]
            if value and value.casefold() not in {q.casefold() for q in queries}:
                queries.append(value)
            if len(queries) >= MAX_QUERIES_PER_REQUIREMENT:
                break
        if not queries:
            queries = [description[:MAX_QUERY_CHARS]]

        entities = []
        for entity in item.get("entities", []) or []:
            value = str(entity or "").strip()[:200]
            if value and value not in entities:
                entities.append(value)

        dependencies = []
        for dependency in item.get("depends_on", []) or []:
            value = str(dependency or "").strip()
            if value and value not in dependencies:
                dependencies.append(value)

        normalized.append({
            "id": requirement_id,
            "kind": kind,
            "type": requirement_type,
            "description": description,
            "queries": queries,
            "entities": entities,
            "depends_on": dependencies,
        })

    by_id = {item["id"]: item for item in normalized}
    indegree = {key: 0 for key in by_id}
    dependents: dict[str, list[str]] = defaultdict(list)
    for item in normalized:
        for dependency in item["depends_on"]:
            if dependency not in by_id:
                raise ValueError(
                    f"requirement {item['id']} depends on unknown requirement {dependency}"
                )
            indegree[item["id"]] += 1
            dependents[dependency].append(item["id"])

    queue = deque(key for key, degree in indegree.items() if degree == 0)
    depth = {key: 1 for key in queue}
    visited = 0
    while queue:
        current = queue.popleft()
        visited += 1
        for child in dependents[current]:
            depth[child] = max(depth.get(child, 1), depth[current] + 1)
            if depth[child] > MAX_DEPENDENCY_DEPTH:
                raise ValueError("requirement dependency depth exceeds configured limit")
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if visited != len(by_id):
        raise ValueError("requirement dependencies contain a cycle")
    return normalized


def retrieval_queries(requirements: list[dict[str, Any]], raw_query: str, limit: int = 3) -> list[str]:
    """Return bounded unique text/either queries, always retaining raw query first."""
    result: list[str] = []
    for value in [raw_query] + [
        query
        for requirement in requirements
        if requirement.get("kind") in {"text", "either"}
        for query in requirement.get("queries", [])
    ]:
        query = str(value or "").strip()
        if query and query.casefold() not in {item.casefold() for item in result}:
            result.append(query)
        if len(result) >= max(1, int(limit)):
            break
    return result
