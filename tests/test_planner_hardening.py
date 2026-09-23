"""
Regression tests for the planner hardening pass.

Run with either:
    python manage.py test tests.test_planner_hardening
    python -m pytest tests/test_planner_hardening.py

No network, no LLM, no database: every LLM/SQL/retrieval call is patched.
"""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest import mock

from planner.services import agent
from planner.services.history import bound_history_for_planner, split_history
from planner.services.sufficiency import (
    LOW_RETRIEVAL_CONFIDENCE,
    NO_DOCUMENT_EVIDENCE,
    TABLE_ZERO_ROWS,
    assess_evidence,
    documents_insufficient,
    insufficiency_note,
)
from planner.tools.conversation_memory import conversation_memory


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def summary_message(text="Earlier we discussed Rahul's leave balance."):
    return {
        "role": "system",
        "kind": "summary",
        "content": (
            "CONVERSATION MEMORY\nThis is a summary of older conversation "
            "context. Treat it as conversational context only.\n\n" + text
        ),
    }


def recent_messages(n, prefix="m"):
    messages = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({"role": role, "content": f"{prefix}{i}"})
    return messages


def rows_result(rows):
    return {
        "success": True,
        "columns": ["name", "salary"],
        "rows": rows,
        "row_count": len(rows),
        "truncated": False,
        "tables_used": ["ws_1_doc_1_table_1"],
    }


SCHEMA = [
    {
        "document_id": 1,
        "document_title": "employees.xlsx",
        "table_name": "ws_1_doc_1_table_1",
        "columns": ["name", "salary"],
        "numeric_columns": ["salary"],
        "row_count": 10,
    }
]

ENTITY_RESOLUTION = {
    "references": [
        {
            "reference": "Rahul",
            "status": "RESOLVED",
            "match": {
                "type": "entity",
                "document_id": 1,
                "value": "Rahul Sharma",
                "locations": [
                    {"table_name": "ws_1_doc_1_table_1", "column": "name"}
                ],
            },
        }
    ],
    "has_ambiguity": False,
    "has_resolved": True,
}


def make_state(capabilities, *, resolution=None, **overrides):
    state = {
        "query": "What is Rahul's salary?",
        "workspace_id": 1,
        "conversation_history": [],
        "relevant_document_ids": [1, 2],
        "query_understanding": {
            "category_hint": "structured",
            "operation": "lookup",
            "references": ["Rahul"],
            "requested_fields": ["salary"],
        },
        "resolved_references": resolution or {"references": []},
        "execution_plan": {
            "status": "OK",
            "capabilities": capabilities,
            "actions": [],
            "clarification_options": [],
        },
    }
    state.update(overrides)
    return state


class AgentTestCase(unittest.TestCase):
    def setUp(self):
        # get_stream_writer() only works inside a running graph.
        patcher = mock.patch.object(agent, "_emit_event")
        patcher.start()
        self.addCleanup(patcher.stop)


# --------------------------------------------------------------------------
# history helper
# --------------------------------------------------------------------------


class HistoryTests(unittest.TestCase):
    def test_summary_survives_and_all_ten_recent_kept(self):
        history = [summary_message()] + recent_messages(10)
        bounded = bound_history_for_planner(history)

        self.assertEqual(bounded[0]["role"], "system")
        self.assertIn("Rahul's leave balance", bounded[0]["content"])
        self.assertEqual(
            [m["content"] for m in bounded[1:]],
            [f"m{i}" for i in range(10)],
        )

    def test_old_slice_would_have_dropped_summary(self):
        history = [summary_message()] + recent_messages(10)
        self.assertNotIn("summary", str(history[-8:]).lower())

    def test_long_messages_truncated_newest_kept(self):
        history = [
            {"role": "user", "content": "x" * 5000},
            {"role": "assistant", "content": "y" * 5000},
        ]
        bounded = bound_history_for_planner(history, max_chars_per_message=100)
        self.assertTrue(all(len(m["content"]) <= 100 for m in bounded))
        self.assertEqual(bounded[-1]["role"], "assistant")

    def test_split_history(self):
        summary, regular = split_history([summary_message()] + recent_messages(3))
        self.assertIsNotNone(summary)
        self.assertEqual(len(regular), 3)


# --------------------------------------------------------------------------
# conversation memory tool
# --------------------------------------------------------------------------


class ConversationMemoryTests(unittest.TestCase):
    def setUp(self):
        self.history = [summary_message()] + [
            {"role": "user", "content": "What is Rahul's salary?"},
            {"role": "assistant", "content": "Rahul earns 90000."},
        ] + recent_messages(8, prefix="filler")

    def test_recap_returns_summary_and_recent(self):
        for query in (
            "What did we discuss earlier?",
            "Summarize what we talked about.",
            "recap please",
        ):
            result = conversation_memory(
                query=query, conversation_history=self.history
            )
            self.assertEqual(result["resolution"], "recap", query)
            self.assertEqual(result["messages"][0]["role"], "system")
            self.assertGreater(result["message_count"], 6)

    def test_answer_you_gave_returns_last_assistant_message(self):
        result = conversation_memory(
            query="What was the answer you gave me?",
            conversation_history=self.history,
        )
        self.assertEqual(result["resolution"], "previous_answer")
        # summary first, then exactly the last assistant message
        self.assertEqual(result["messages"][0]["role"], "system")
        self.assertEqual(result["messages"][-1]["role"], "assistant")

    def test_summary_returned_for_normal_followup(self):
        result = conversation_memory(
            query="What about his leave?", conversation_history=self.history
        )
        self.assertTrue(result.get("has_summary"))
        self.assertEqual(result["messages"][0]["role"], "system")

    def test_no_summary_no_crash(self):
        result = conversation_memory(
            query="What did we discuss earlier?",
            conversation_history=recent_messages(4),
        )
        self.assertEqual(result["resolution"], "recap")
        self.assertEqual(result["message_count"], 4)

    def test_empty_history(self):
        result = conversation_memory(query="hello there", conversation_history=[])
        self.assertEqual(result["message_count"], 0)


# --------------------------------------------------------------------------
# sufficiency
# --------------------------------------------------------------------------


class SufficiencyTests(unittest.TestCase):
    def test_zero_rows_is_insufficient(self):
        out = assess_evidence(
            wants_table=True,
            table_result_status="NOT_FOUND",
            wants_documents=False,
            document_chunk_count=0,
            retrieval_confidence=0.0,
            min_retrieval_confidence=None,
        )
        self.assertFalse(out["sufficient"])
        self.assertEqual(out["reasons"], [TABLE_ZERO_ROWS])

    def test_success_is_sufficient(self):
        out = assess_evidence(
            wants_table=True,
            table_result_status="SUCCESS",
            wants_documents=True,
            document_chunk_count=3,
            retrieval_confidence=0.9,
            min_retrieval_confidence=0.5,
        )
        self.assertTrue(out["sufficient"])

    def test_threshold_only_applies_when_configured(self):
        self.assertFalse(documents_insufficient(3, 0.01, None))
        self.assertTrue(documents_insufficient(3, 0.01, 0.5))
        self.assertTrue(documents_insufficient(0, 0.99, None))

    def test_notes(self):
        self.assertIsNone(insufficiency_note([TABLE_ZERO_ROWS]))
        self.assertEqual(
            insufficiency_note([NO_DOCUMENT_EVIDENCE])["result_status"],
            "INSUFFICIENT_EVIDENCE",
        )
        self.assertIn(
            "low relevance", insufficiency_note([LOW_RETRIEVAL_CONFIDENCE])["content"]
        )


# --------------------------------------------------------------------------
# dispatch: evidence budget, recovery, metadata
# --------------------------------------------------------------------------


class EvidenceBudgetTests(AgentTestCase):
    def test_table_result_survives_when_retrieval_returns_eight_chunks(self):
        chunks = [
            {"content": f"chunk {i}", "document_id": 2} for i in range(8)
        ]
        state = make_state(
            ["TABLE_QUERY", "DOCUMENT_SEARCH"], resolution=ENTITY_RESOLUTION
        )

        with mock.patch.object(agent, "get_workspace_schema", return_value=SCHEMA), \
             mock.patch.object(
                 agent, "generate_sql",
                 return_value={"needs_sql": True, "sql": "SELECT 1", "parameters": []},
             ), \
             mock.patch.object(
                 agent, "query_tables",
                 return_value=rows_result([{"name": "Rahul Sharma", "salary": "90000"}]),
             ), \
             mock.patch.object(
                 agent, "search_evidence",
                 return_value={"error_code": None, "error": None,
                               "evidence": chunks, "confidence": 0.9,
                               "candidate_count": 8},
             ):
            update = agent.dispatch_node(state)

        evidence = update["evidence"]
        self.assertLessEqual(len(evidence), agent.MAX_EVIDENCE)
        self.assertEqual(evidence[0]["source_type"], "table_query")
        self.assertEqual(evidence[0]["result_status"], "SUCCESS")


class TableRecoveryTests(AgentTestCase):
    def dispatch(self, sql_side_effect, query_side_effect, resolution=ENTITY_RESOLUTION):
        state = make_state(["TABLE_QUERY"], resolution=resolution)
        gen = mock.Mock(side_effect=sql_side_effect)
        qt = mock.Mock(side_effect=query_side_effect)
        with mock.patch.object(agent, "get_workspace_schema", return_value=SCHEMA), \
             mock.patch.object(agent, "generate_sql", gen), \
             mock.patch.object(agent, "query_tables", qt):
            update = agent.dispatch_node(state)
        return update, gen, qt

    SQL = {"needs_sql": True, "sql": "SELECT * FROM t", "parameters": []}

    def test_sql_error_is_repaired_once(self):
        update, gen, qt = self.dispatch(
            [self.SQL, {**self.SQL, "sql": "SELECT * FROM fixed"}],
            [ValueError("no such column: nme"),
             rows_result([{"name": "Rahul Sharma", "salary": "90000"}])],
        )
        self.assertEqual(gen.call_count, 2)
        self.assertEqual(update["evidence"][0]["result_status"], "SUCCESS")
        self.assertEqual(update["recovery_attempts"], 1)
        repair_kwargs = gen.call_args_list[1].kwargs
        self.assertIn("no such column", repair_kwargs["previous_attempt"]["error"])

    def test_zero_rows_for_known_entity_retries_exactly_once_then_not_found(self):
        update, gen, qt = self.dispatch(
            [self.SQL, self.SQL, self.SQL],  # a 3rd call would be a bug
            [rows_result([]), rows_result([]), rows_result([])],
        )
        self.assertEqual(gen.call_count, 2)
        self.assertEqual(qt.call_count, 2)
        self.assertEqual(update["evidence"][0]["result_status"], "NOT_FOUND")
        self.assertIn("TABLE_ZERO_ROWS", update["insufficient_reasons"])
        self.assertTrue(gen.call_args_list[1].kwargs["previous_attempt"]["zero_rows"])

    def test_zero_rows_without_resolved_entity_does_not_retry(self):
        update, gen, qt = self.dispatch(
            [self.SQL], [rows_result([])], resolution={"references": []}
        )
        self.assertEqual(gen.call_count, 1)
        self.assertEqual(update["evidence"][0]["result_status"], "NOT_FOUND")
        self.assertEqual(update["recovery_attempts"], 0)

    def test_two_failures_degrade_instead_of_crashing(self):
        update, gen, qt = self.dispatch(
            [self.SQL, self.SQL],
            [ValueError("boom"), ValueError("boom again")],
        )
        self.assertFalse(update["tool_failed"])
        self.assertEqual(update["evidence"][0]["result_status"], "RETRIEVAL_ERROR")
        # internal error text must not reach the answer model
        self.assertNotIn("boom", json.dumps(update["evidence"]))

        with mock.patch.object(agent, "generate_answer") as answer_llm:
            final = agent.finish_node({**make_state(["TABLE_QUERY"]), **update})
        answer_llm.assert_not_called()
        self.assertEqual(final["route"], "ERROR")
        self.assertEqual(final["error_code"], "RETRIEVAL_ERROR")

    def test_invalid_generator_output_is_recoverable(self):
        update, gen, qt = self.dispatch(
            [json.JSONDecodeError("bad", "x", 0), self.SQL],
            [rows_result([{"name": "Rahul Sharma", "salary": "1"}])],
        )
        self.assertEqual(gen.call_count, 2)
        self.assertEqual(update["evidence"][0]["result_status"], "SUCCESS")

    def test_needs_sql_false_returns_no_table_evidence(self):
        update, gen, qt = self.dispatch(
            [{"needs_sql": False, "sql": None, "parameters": []}], []
        )
        qt.assert_not_called()
        self.assertEqual(update["evidence"], [])

    def test_infrastructure_error_is_not_retried(self):
        update, gen, qt = self.dispatch(
            [self.SQL], [ValueError("SQLite database not found: /x")]
        )
        self.assertEqual(gen.call_count, 1)
        self.assertTrue(update["tool_failed"])
        self.assertEqual(update["error_code"], "DATABASE_ERROR")


class DocumentRecoveryTests(AgentTestCase):
    def search(self, results, capabilities=("DOCUMENT_SEARCH",), **kw):
        state = make_state(list(capabilities), **kw)
        state["query_understanding"] = {
            "operation": "answer", "references": [], "requested_fields": [],
        }
        state["query"] = "what is the refund policy"
        fn = mock.Mock(side_effect=results)
        with mock.patch.object(agent, "search_evidence", fn):
            update = agent.dispatch_node(state)
        return update, fn

    @staticmethod
    def hit(n, confidence=0.9):
        return {
            "error_code": None, "error": None, "confidence": confidence,
            "candidate_count": n,
            "evidence": [{"content": f"c{i}", "document_id": 1} for i in range(n)],
        }

    def test_no_chunks_triggers_one_broadened_retry(self):
        update, fn = self.search([self.hit(0), self.hit(2)])
        self.assertEqual(fn.call_count, 2)
        self.assertEqual(len(update["evidence"]), 2)
        self.assertTrue(update["evidence_sufficient"])
        self.assertEqual(fn.call_args_list[1].kwargs["query"], "what is the refund policy")

    def test_still_empty_after_retry_leaves_no_evidence(self):
        update, fn = self.search([self.hit(0), self.hit(0)])
        self.assertEqual(fn.call_count, 2)
        self.assertEqual(update["evidence"], [])
        self.assertFalse(update["evidence_sufficient"])
        final = agent.finish_node({**make_state(["DOCUMENT_SEARCH"]), **update})
        self.assertEqual(final["route"], "NO_EVIDENCE")

    def test_retrieval_error_alone_is_a_real_failure(self):
        update, fn = self.search(
            [{"error_code": "INDEX_MISSING", "error": "x", "evidence": [],
              "confidence": 0.0, "candidate_count": 0}]
        )
        self.assertTrue(update["tool_failed"])
        self.assertEqual(update["error_code"], "INDEX_MISSING")

    def test_low_confidence_retry_only_when_threshold_configured(self):
        with mock.patch.object(agent, "MIN_RETRIEVAL_CONFIDENCE", None):
            update, fn = self.search([self.hit(3, confidence=0.01)])
        self.assertEqual(fn.call_count, 1)

        with mock.patch.object(agent, "MIN_RETRIEVAL_CONFIDENCE", 0.5):
            update, fn = self.search([self.hit(3, 0.01), self.hit(3, 0.02)])
        self.assertEqual(fn.call_count, 2)
        self.assertIn("LOW_RETRIEVAL_CONFIDENCE", update["insufficient_reasons"])
        self.assertEqual(update["evidence"][-1]["result_status"], "INSUFFICIENT_EVIDENCE")

    def test_table_success_plus_no_docs_adds_insufficient_note(self):
        state = make_state(
            ["TABLE_QUERY", "DOCUMENT_SEARCH"], resolution=ENTITY_RESOLUTION
        )
        with mock.patch.object(agent, "get_workspace_schema", return_value=SCHEMA), \
             mock.patch.object(agent, "generate_sql",
                               return_value={"needs_sql": True, "sql": "S", "parameters": []}), \
             mock.patch.object(agent, "query_tables",
                               return_value=rows_result([{"name": "R", "salary": "1"}])), \
             mock.patch.object(agent, "search_evidence",
                               side_effect=[self.hit(0), self.hit(0)]):
            update = agent.dispatch_node(state)
        statuses = [e["result_status"] for e in update["evidence"]]
        self.assertEqual(statuses, ["SUCCESS", "INSUFFICIENT_EVIDENCE"])


class SufficiencyGateTests(AgentTestCase):
    """finish_node must honour evidence_sufficient / answerable."""

    @staticmethod
    def hit(n, confidence):
        return {
            "error_code": None, "error": None, "confidence": confidence,
            "candidate_count": n,
            "evidence": [{"content": f"c{i}", "document_id": 1} for i in range(n)],
        }

    def run_docs_only(self, results):
        state = make_state(["DOCUMENT_SEARCH"])
        state["query"] = "what is the refund policy"
        state["query_understanding"] = {
            "operation": "answer", "references": [], "requested_fields": [],
        }
        with mock.patch.object(agent, "MIN_RETRIEVAL_CONFIDENCE", 0.5), \
             mock.patch.object(agent, "search_evidence", side_effect=results):
            update = agent.dispatch_node(state)
        return state, update

    def test_weak_documents_only_never_reach_answer_llm(self):
        state, update = self.run_docs_only([self.hit(3, 0.01), self.hit(3, 0.02)])
        self.assertFalse(update["answerable"])
        self.assertFalse(update["evidence_sufficient"])
        with mock.patch.object(agent, "generate_answer") as llm:
            final = agent.finish_node({**state, **update})
        llm.assert_not_called()
        self.assertEqual(final["route"], "NO_EVIDENCE")
        self.assertEqual(final["error_code"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(final["final_evidence"], [])

    def test_strong_documents_are_answered(self):
        state, update = self.run_docs_only([self.hit(3, 0.9)])
        self.assertTrue(update["answerable"])
        with mock.patch.object(agent, "generate_answer", return_value="ok") as llm:
            final = agent.finish_node({**state, **update})
        llm.assert_called_once()
        self.assertEqual(final["route"], "ANSWER")

    def test_mixed_request_keeps_partial_answer_when_documents_weak(self):
        state = make_state(
            ["TABLE_QUERY", "DOCUMENT_SEARCH"], resolution=ENTITY_RESOLUTION
        )
        with mock.patch.object(agent, "MIN_RETRIEVAL_CONFIDENCE", 0.5), \
             mock.patch.object(agent, "get_workspace_schema", return_value=SCHEMA), \
             mock.patch.object(agent, "generate_sql",
                               return_value={"needs_sql": True, "sql": "S", "parameters": []}), \
             mock.patch.object(agent, "query_tables",
                               return_value=rows_result([{"name": "R", "salary": "1"}])), \
             mock.patch.object(agent, "search_evidence",
                               side_effect=[self.hit(3, 0.01), self.hit(3, 0.02)]):
            update = agent.dispatch_node(state)
        self.assertTrue(update["answerable"])
        self.assertFalse(update["evidence_sufficient"])
        self.assertEqual(update["evidence"][-1]["result_status"], "INSUFFICIENT_EVIDENCE")
        with mock.patch.object(agent, "generate_answer", return_value="partial") as llm:
            final = agent.finish_node({**state, **update})
        llm.assert_called_once()
        self.assertEqual(final["route"], "ANSWER")


class DocumentScopeTests(AgentTestCase):
    @staticmethod
    def empty():
        return {"error_code": None, "error": None, "evidence": [],
                "confidence": 0.0, "candidate_count": 0}

    @staticmethod
    def base_state(resolution):
        state = make_state(["DOCUMENT_SEARCH"], resolution=resolution)
        state["query"] = "what does the salary sheet say about deductions"
        state["query_understanding"] = {
            "operation": "answer", "references": ["salary sheet"],
            "requested_fields": [],
        }
        state["relevant_document_ids"] = [1, 2, 3, 4]
        return state

    def test_recovery_stays_inside_explicitly_resolved_document(self):
        resolution = {"references": [{
            "reference": "salary sheet", "status": "RESOLVED",
            "match": {"type": "document", "document_id": 3, "title": "salary_sheet.xlsx"},
        }]}
        fn = mock.Mock(side_effect=[self.empty(), self.empty()])
        with mock.patch.object(agent, "search_evidence", fn):
            agent.dispatch_node(self.base_state(resolution))
        self.assertEqual(fn.call_count, 2)
        for call in fn.call_args_list:
            self.assertEqual(call.kwargs["allowed_document_ids"], [3])
        # only the QUERY was broadened
        self.assertEqual(
            fn.call_args_list[1].kwargs["query"],
            "what does the salary sheet say about deductions",
        )

    def test_unscoped_search_may_broaden_to_whole_workspace(self):
        state = self.base_state({"references": []})
        state["query_understanding"]["references"] = []
        fn = mock.Mock(side_effect=[self.empty(), self.empty()])
        with mock.patch.object(agent, "search_evidence", fn):
            agent.dispatch_node(state)
        for call in fn.call_args_list:
            self.assertEqual(call.kwargs["allowed_document_ids"], [1, 2, 3, 4])

    def test_entity_match_does_not_scope_text_search(self):
        # Rahul lives in a spreadsheet (doc 1); the policy text is elsewhere.
        state = self.base_state(ENTITY_RESOLUTION)
        fn = mock.Mock(return_value={
            "error_code": None, "error": None, "confidence": 0.9,
            "candidate_count": 1,
            "evidence": [{"content": "policy", "document_id": 2}]})
        with mock.patch.object(agent, "search_evidence", fn):
            agent.dispatch_node(state)
        self.assertEqual(fn.call_args.kwargs["allowed_document_ids"], [1, 2, 3, 4])


class SourceMetadataWiringTests(AgentTestCase):
    def test_resolved_document_passes_document_id(self):
        resolution = {"references": [{
            "reference": "employees", "status": "RESOLVED",
            "match": {"type": "document", "document_id": 7,
                      "title": "employees.xlsx"},
        }]}
        state = make_state(["SOURCE_METADATA"], resolution=resolution)
        tool = mock.Mock(return_value={
            "status": "FOUND", "source_type": "workspace_metadata",
            "sources": [], "content": {"document_count": 1, "documents": []},
        })
        with mock.patch.object(agent, "source_metadata", tool):
            update = agent.dispatch_node(state)
        self.assertEqual(tool.call_args.kwargs["document_id"], 7)
        self.assertEqual(update["evidence"][0]["result_status"], "SUCCESS")

    def test_tool_not_found_is_preserved(self):
        resolution = {"references": [{
            "reference": "x", "status": "RESOLVED",
            "match": {"type": "document", "document_id": 7, "title": "x.xlsx"},
        }]}
        state = make_state(["SOURCE_METADATA"], resolution=resolution)
        tool = mock.Mock(return_value={
            "status": "NOT_FOUND", "sources": [],
            "content": {"document_count": 0, "documents": []},
        })
        with mock.patch.object(agent, "source_metadata", tool):
            update = agent.dispatch_node(state)
        self.assertEqual(update["evidence"][0]["result_status"], "NOT_FOUND")
        with mock.patch.object(agent, "generate_answer", return_value="not found"):
            final = agent.finish_node({**state, **update})
        self.assertEqual(final["route"], "NOT_FOUND")

    def test_unresolved_reference_is_not_answered_with_whole_catalog(self):
        resolution = {"references": [{
            "reference": "salary_2024.xlsx", "status": "NOT_FOUND", "matches": [],
        }]}
        state = make_state(["SOURCE_METADATA"], resolution=resolution)
        catalog = {
            "status": "FOUND",
            "content": {"documents": [{"title": "employees.xlsx"},
                                      {"title": "policy.pdf"}]},
        }
        with mock.patch.object(agent, "source_metadata", return_value=catalog):
            update = agent.dispatch_node(state)
        item = update["evidence"][0]
        self.assertEqual(item["result_status"], "NOT_FOUND")
        self.assertEqual(item["requested_reference"], ["salary_2024.xlsx"])
        self.assertEqual(item["content"]["documents"], [])
        self.assertEqual(
            item["content"]["available_documents"], ["employees.xlsx", "policy.pdf"]
        )
        with mock.patch.object(agent, "generate_answer", return_value="not found") as llm:
            final = agent.finish_node({**state, **update})
        self.assertEqual(final["route"], "NOT_FOUND")
        llm.assert_called_once()

    def test_listing_with_no_references_returns_catalog(self):
        state = make_state(["SOURCE_METADATA"], resolution={"references": []})
        catalog = {"status": "FOUND", "content": {"documents": [{"title": "a"}]}}
        with mock.patch.object(agent, "source_metadata", return_value=catalog):
            update = agent.dispatch_node(state)
        self.assertEqual(update["evidence"][0]["result_status"], "SUCCESS")


# --------------------------------------------------------------------------
# whole-request behaviour
# --------------------------------------------------------------------------


class RequestBoundTests(AgentTestCase):
    def test_worst_case_uses_at_most_four_llm_calls(self):
        """planner + SQL + SQL repair + answer. Doc retry must NOT also run."""
        calls = {"planner": 0, "sql": 0, "answer": 0}

        def planner(**kwargs):
            calls["planner"] += 1
            payload = {
                "category_hint": "mixed", "operation": "calculate",
                "references": ["Rahul"], "requested_fields": ["salary"],
                "reason": "x",
            }
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=json.dumps(payload)))])

        def sql(**kwargs):
            calls["sql"] += 1
            return {"needs_sql": True, "sql": "SELECT 1", "parameters": []}

        def answer(**kwargs):
            calls["answer"] += 1
            return "Not found."

        state = make_state([], resolution=None, query="Calculate Rahul's deduction")
        state.pop("execution_plan"); state.pop("query_understanding")
        state.pop("resolved_references")

        search = mock.Mock(return_value={
            "error_code": None, "error": None, "evidence": [],
            "confidence": 0.0, "candidate_count": 0})

        with mock.patch.object(agent.router, "generate", side_effect=planner), \
             mock.patch.object(agent, "resolve_references", return_value=ENTITY_RESOLUTION), \
             mock.patch.object(agent, "get_workspace_schema", return_value=SCHEMA), \
             mock.patch.object(agent, "generate_sql", side_effect=sql), \
             mock.patch.object(agent, "query_tables", return_value=rows_result([])), \
             mock.patch.object(agent, "search_evidence", search), \
             mock.patch.object(agent, "generate_answer", side_effect=answer):
            state.update(agent.agent_node(state))
            state.update(agent.dispatch_node(state))
            state.update(agent.finish_node(state))

        total = calls["planner"] + calls["sql"] + calls["answer"]
        self.assertLessEqual(total, 4, calls)
        self.assertEqual(calls["sql"], 2)
        self.assertEqual(search.call_count, 1)  # recovery already spent on SQL
        self.assertEqual(state["recovery_attempts"], 1)

    def test_chitchat_makes_no_llm_calls(self):
        state = make_state([], query="Hi!")
        with mock.patch.object(agent.router, "generate") as planner, \
             mock.patch.object(agent, "search_evidence") as search:
            update = agent.agent_node(state)
        planner.assert_not_called()
        search.assert_not_called()
        self.assertEqual(update["route"], "CHITCHAT")
        self.assertEqual(agent.route_after_agent({**state, **update}), "finish")
        final = agent.finish_node({**state, **update})
        self.assertEqual(final["route"], "CHITCHAT")
        self.assertTrue(final["answer"])

    def test_real_question_is_not_chitchat(self):
        self.assertIsNone(agent._chitchat_reply("hi, what is Rahul's salary?"))
        self.assertIsNotNone(agent._chitchat_reply("Thank you"))

    def test_answer_llm_failure_is_graceful(self):
        state = make_state(["PRODUCT_KNOWLEDGE"], query="What is QueryPilot?")
        update = agent.dispatch_node(state)
        with mock.patch.object(agent, "generate_answer", side_effect=RuntimeError("429")):
            final = agent.finish_node({**state, **update})
        self.assertEqual(final["route"], "ERROR")
        self.assertEqual(final["error_code"], "ANSWER_GENERATION_ERROR")
        self.assertNotIn("429", final["answer"])
        self.assertTrue(final["final_evidence"])

    def test_deadline_exceeded_stops_before_tools(self):
        state = make_state(["PRODUCT_KNOWLEDGE"], deadline_monotonic=1.0)  # long past
        update = agent.dispatch_node(state)
        self.assertTrue(update["tool_failed"])
        self.assertEqual(update["error_code"], "REQUEST_TIMEOUT")
        final = agent.finish_node({**state, **update})
        self.assertIn("took longer", final["answer"])

    def test_deadline_exceeded_skips_recovery(self):
        state = make_state(["TABLE_QUERY"], resolution=ENTITY_RESOLUTION)
        # make the deadline pass only after dispatch starts
        checks = iter([False, True, True, True, True])
        gen = mock.Mock(return_value={"needs_sql": True, "sql": "S", "parameters": []})
        with mock.patch.object(agent, "_past_deadline", side_effect=lambda s: next(checks, True)), \
             mock.patch.object(agent, "get_workspace_schema", return_value=SCHEMA), \
             mock.patch.object(agent, "generate_sql", gen), \
             mock.patch.object(agent, "query_tables", return_value=rows_result([])):
            update = agent.dispatch_node(state)
        self.assertEqual(gen.call_count, 1)
        self.assertEqual(update["evidence"][0]["result_status"], "NOT_FOUND")

    def test_planner_context_contains_summary(self):
        state = make_state(
            [], conversation_history=[summary_message()] + recent_messages(10)
        )
        context = json.loads(agent._build_context(state))
        history = context["conversation_history"]
        self.assertIn("Rahul's leave balance", history[0]["content"])
        self.assertEqual(len(history), 11)


if __name__ == "__main__":
    unittest.main()