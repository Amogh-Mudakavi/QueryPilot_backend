from __future__ import annotations

import json
import logging
from typing import Any

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from llm.router import router

from planner.prompts.agent import ACTION_SCHEMA, SYSTEM_PROMPT
from planner.services.answer import generate_answer
from planner.services.schema_registry import get_workspace_schema
from planner.services.sql_generation import generate_sql
from planner.services.state import AgentState

from planner.tools.conversation_memory import conversation_memory
from planner.tools.query_tables import query_tables
from planner.tools.search_evidence import search_evidence


logger = logging.getLogger(__name__)

MAX_ITERATIONS = 3
MAX_TOOL_CALLS = 4
MAX_STATE_EVIDENCE = 8


class AgentRuntimeError(Exception):
    pass


def build_agent_graph():
    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("dispatch", dispatch_node)
    graph.add_node("finish", finish_node)

    graph.add_edge(START, "agent")

    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "dispatch": "dispatch",
            "finish": "finish",
        },
    )

    graph.add_conditional_edges(
        "dispatch",
        route_after_dispatch,
        {
            "agent": "agent",
            "finish": "finish",
        },
    )

    graph.add_edge("finish", END)

    return graph.compile()


def _emit_event(
    *,
    event: str,
    status: str,
    message: str,
    data: Any = None,
):
    payload = {
        "event": event,
        "status": status,
        "message": message,
    }

    if data is not None:
        payload["data"] = data

    get_stream_writer()(payload)


def agent_node(
    state: AgentState,
) -> dict[str, Any]:

    if state.get("iteration_count", 0) >= MAX_ITERATIONS:
        return {
            "error": "Maximum agent iterations reached.",
            "evidence_sufficient": False,
            "tool_failed": True,
        }

    _emit_event(
        event="planning",
        status="started",
        message="Deciding what evidence is required.",
    )

    response = router.generate(
        task="planner",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": _build_agent_context(state),
            },
        ],
        temperature=0.0,
        max_tokens=400,
        response_format=ACTION_SCHEMA,
        include_reasoning=False,
    )

    action = _parse_action(response)

    _emit_event(
        event="planning",
        status="completed",
        message="Evidence mode selected.",
        data={
            "mode": action["mode"],
        },
    )

    return {
        "tool_history": state.get(
            "tool_history",
            [],
        )
        + [
            {
                "stage": "agent_decision",
                "action": action,
            }
        ],
        "iteration_count": (
            state.get("iteration_count", 0)
            + 1
        ),
        "_pending_action": action,
    }


def dispatch_node(
    state: AgentState,
) -> dict[str, Any]:

    if state.get("tool_failed"):
        return {
            "error": state.get(
                "error",
                "Previous step failed.",
            ),
            "evidence_sufficient": False,
            "tool_failed": True,
        }

    action = state.get("_pending_action")

    if not action:
        return _failure(
            "Agent produced no pending action."
        )

    total_tool_calls = state.get(
        "total_tool_calls",
        0,
    )

    if total_tool_calls >= MAX_TOOL_CALLS:
        return _failure(
            "Maximum tool-call limit reached."
        )

    mode = action["mode"]
    total_tool_calls += 1

    raw_query = state.get(
        "query",
        "",
    )

    allowed_document_ids = (
        state.get(
            "relevant_document_ids",
            [],
        )
    )

    table_query = raw_query
    text_query = raw_query

    if mode == "HYBRID":

        table_query = (
            action.get("table_query")
            or raw_query
        ).strip()

        text_query = (
            action.get("text_query")
            or raw_query
        ).strip()

    logger.info(
        "agent.dispatch: mode=%s text_query=%s table_query=%s relevant_document_ids=%s",
        mode,
        text_query,
        table_query,
        allowed_document_ids,
    )

    try:
        result = {
            "evidence": list(
                state.get(
                    "evidence",
                    [],
                )
            ),
        }

        # ---------------------------------------------------------
        # TEXT
        # ---------------------------------------------------------

        if mode in {"TEXT", "HYBRID"}:

            text_result = _run_text(
                state,
                query=text_query,
            )

            logger.info(
                "agent.dispatch: text_result candidate_count=%s evidence=%d",
                text_result.get(
                    "candidate_count"
                ),
                len(
                    text_result.get(
                        "evidence",
                        [],
                    )
                ),
            )

            result["evidence"] = (
                result["evidence"]
                + text_result.get(
                    "evidence",
                    [],
                )
            )

        # ---------------------------------------------------------
        # TABLE
        # ---------------------------------------------------------

        if mode in {"TABLE", "HYBRID"}:

            table_result = _run_table(
                state,
                query=table_query,
            )

            table_evidence = table_result.get(
                "evidence",
                [],
            )

            table_rows = None
            tables_used = None

            if table_evidence:
                content = table_evidence[0].get(
                    "content",
                    {},
                )

                if isinstance(content, dict):
                    table_rows = content.get(
                        "row_count"
                    )
                    tables_used = content.get(
                        "tables_used"
                    )

            logger.info(
                "agent.dispatch: table_result rows=%s tables_used=%s",
                table_rows,
                tables_used,
            )

            result["evidence"] = (
                result["evidence"]
                + table_evidence
            )

        # ---------------------------------------------------------
        # MEMORY
        # ---------------------------------------------------------

        if mode == "CONVERSATION_MEMORY":
            result = _run_memory(state)

        # ---------------------------------------------------------
        # FINISH
        # ---------------------------------------------------------

        if mode == "FINISH":
            return {
                "evidence_sufficient": bool(
                    state.get("evidence")
                ),
                "total_tool_calls": total_tool_calls,
                "tool_failed": False,
            }

    except Exception as exc:

        _emit_event(
            event="agent_error",
            status="failed",
            message=f"Evidence step failed: {mode}.",
            data={
                "error": str(exc),
            },
        )

        return {
            "error": str(exc),
            "evidence_sufficient": False,
            "total_tool_calls": total_tool_calls,
            "last_tool": mode,
            "tool_failed": True,
        }

    return {
        **result,
        "last_tool": mode,
        "total_tool_calls": total_tool_calls,
        "tool_failed": False,
        "tool_history": state.get(
            "tool_history",
            [],
        )
        + [
            {
                "stage": "evidence_execution",
                "mode": mode,
                "table_query": (
                    table_query
                    if mode in {"TABLE", "HYBRID"}
                    else None
                ),
                "text_query": (
                    text_query
                    if mode in {"TEXT", "HYBRID"}
                    else None
                ),
            }
        ],
    }


def finish_node(
    state: AgentState,
) -> dict[str, Any]:

    if state.get("tool_failed"):
        return {
            "final_evidence": state.get(
                "evidence",
                [],
            ),
            "evidence_sufficient": False,
            "answer": (
                "I could not complete that request because "
                "the required evidence step failed."
            ),
            "grounding_passed": None,
            "error": state.get(
                "error",
                "",
            ),
        }

    evidence = state.get(
        "evidence",
        [],
    )

    if not evidence:
        return {
            "final_evidence": [],
            "evidence_sufficient": False,
            "answer": (
                "I could not find sufficient information "
                "in the uploaded documents."
            ),
            "grounding_passed": None,
            "error": "",
        }

    _emit_event(
        event="generating_answer",
        status="started",
        message="Generating the final answer.",
    )

    answer = generate_answer(
        query=state.get(
            "query",
            "",
        ),
        evidence=evidence,
        conversation_history=state.get(
            "conversation_history",
            [],
        ),
    )

    _emit_event(
        event="generating_answer",
        status="completed",
        message="Final answer generated.",
    )

    return {
        "final_evidence": evidence,
        "evidence_sufficient": True,
        "answer": answer,
        "grounding_passed": None,
        "error": "",
    }


def route_after_agent(
    state: AgentState,
) -> str:

    if state.get("error"):
        return "finish"

    action = state.get("_pending_action")

    if not action:
        return "finish"

    if action["mode"] == "FINISH":
        return "finish"

    if state.get(
        "iteration_count",
        0,
    ) >= MAX_ITERATIONS:
        return "finish"

    return "dispatch"


def route_after_dispatch(
    state: AgentState,
) -> str:

    if state.get("tool_failed"):
        return "finish"

    if state.get("error"):
        return "finish"

    return "finish"


def _build_agent_context(
    state: AgentState,
) -> str:

    payload = {
        "user_query": state.get(
            "query",
            "",
        ),
        "workspace_id": state.get(
            "workspace_id"
        ),
        "relevant_document_ids": state.get(
            "relevant_document_ids",
            [],
        ),
        "evidence": state.get(
            "evidence",
            [],
        ),
        "conversation_history": state.get(
            "conversation_history",
            [],
        ),
        "tool_history": state.get(
            "tool_history",
            [],
        ),
    }

    return json.dumps(
        payload,
        ensure_ascii=False,
        default=str,
    )


def _parse_action(
    response,
) -> dict[str, Any]:

    try:
        content = (
            response
            .choices[0]
            .message
            .content
        )
    except (
        AttributeError,
        IndexError,
    ) as exc:
        raise AgentRuntimeError(
            "Planner returned an invalid response."
        ) from exc

    try:
        action = json.loads(content)
    except json.JSONDecodeError as exc:
        raise AgentRuntimeError(
            "Planner returned invalid JSON."
        ) from exc

    if action.get("mode") not in {
        "TEXT",
        "TABLE",
        "HYBRID",
        "CONVERSATION_MEMORY",
        "FINISH",
    }:
        raise AgentRuntimeError(
            "Planner returned an unsupported evidence mode."
        )

    return action


def _run_text(
    state: AgentState,
    *,
    query: str,
) -> dict[str, Any]:

    allowed_document_ids = state.get(
        "relevant_document_ids",
        [],
    )

    result = search_evidence(
        query=query,
        workspace_id=state["workspace_id"],
        allowed_document_ids=allowed_document_ids,
    )

    evidence = [
        item
        for item in result.get(
            "evidence",
            [],
        )
        if item.get("source_type") == "text_chunk"
    ]

    return {
        "evidence": evidence[
            :MAX_STATE_EVIDENCE
        ],
        "retrieval_confidence": result.get(
            "confidence",
            0.0,
        ),
        "candidate_count": result.get(
            "candidate_count",
            0,
        ),
    }


def _run_table(
    state: AgentState,
    *,
    query: str,
) -> dict[str, Any]:

    allowed_document_ids = (
        state.get(
            "relevant_document_ids",
            [],
        )
    )

    schema = get_workspace_schema(
        state["workspace_id"],
        allowed_document_ids=allowed_document_ids,
    )

    if not schema:
        raise AgentRuntimeError(
            "No tables are available in the allowed documents."
        )

    decision = generate_sql(
        query=query,
        schema=schema,
    )

    if not decision.get("needs_sql"):
        return {
            "evidence": []
        }

    sql = decision.get("sql")

    if not sql:
        raise AgentRuntimeError(
            "SQL generator returned no SQL."
        )

    schema_by_table = {
        item["table_name"]: item
        for item in schema
    }

    allowed_table_names = set(
        schema_by_table.keys()
    )

    result = query_tables(
        workspace_id=state["workspace_id"],
        sql=sql,
        parameters=decision.get(
            "parameters",
            [],
        ),
        allowed_table_names=allowed_table_names,
    )

    used_tables = result.get(
        "tables_used",
        [],
    )

    sources = [
        {
            "document_id": schema_by_table[
                table_name
            ].get("document_id"),
            "table_name": table_name,
        }
        for table_name in used_tables
        if table_name in schema_by_table
    ]

    return {
        "evidence": [
            {
                "source_type": "table_query",
                "workspace_id": state[
                    "workspace_id"
                ],
                "sources": sources,
                "content": result,
            }
        ]
    }


def _run_memory(
    state: AgentState,
) -> dict[str, Any]:

    result = conversation_memory(
        query=state.get(
            "query",
            "",
        ),
        conversation_history=state.get(
            "conversation_history",
            [],
        ),
    )

    return {
        "conversation_history": result.get(
            "messages",
            [],
        ),
        "evidence": list(
            state.get(
                "evidence",
                [],
            )
        ),
    }


def _failure(
    message: str,
    total_tool_calls: int = 0,
):
    return {
        "error": message,
        "evidence_sufficient": False,
        "total_tool_calls": total_tool_calls,
        "tool_failed": True,
    }


agent_graph = build_agent_graph()