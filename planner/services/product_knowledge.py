"""Services Product Knowledge module for QueryPilot."""

from __future__ import annotations

DEVELOPER_FACTS: dict[str, str] = {
    "name": "Amogh Mudakavi",
    "education": (
        "Amogh Mudakavi completed his B.Tech in Electronics and Communication "
        "Engineering (ECE) at PES University, Bengaluru."
    ),
    "why_querypilot_was_built": (
        "Amogh built QueryPilot to create a production-oriented AI application "
        "that combines backend engineering, RAG, structured data querying, "
        "LLM orchestration, document processing, and secure data access in one system."
    ),
    "technical_focus": (
        "His primary technical focus is backend engineering and AI engineering, "
        "with an emphasis on building reliable, scalable systems rather than "
        "frontend-heavy applications."
    ),
    "skills": (
        "His technical skills include Python, Django, Django REST Framework, "
        " MySQL, PostgreSQL, MongoDB, Redis, "
        "Docker, AWS, vector databases, RAG systems, embeddings, "
        "hybrid retrieval, reranking, and LLM application development."
    ),
    "querypilot_focus": (
        "QueryPilot reflects his interest in production-grade AI systems, especially "
        "RAG pipelines, document intelligence, structured data access, LLM agents, "
        "retrieval systems, security boundaries, and backend architecture."
    ),
    "career_goal": (
        "Amogh wants to build a career in backend and AI engineering, with a long-term "
        "goal of working on strong product engineering and machine-learning systems."
    ),
    "project_goal": (
        "QueryPilot was built as a serious engineering project rather than a simple "
        "LLM demo, with emphasis on correctness, security, retrieval quality, "
        "structured-data handling, reliability, and production-style architecture."
    ),
}

PRODUCT_KNOWLEDGE = """\
WHAT QUERYPILOT IS

QueryPilot is a workspace-based AI assistant. A workspace is a closed
knowledge boundary: it contains exactly the documents and structured
tables the user has uploaded.

QueryPilot answers questions using only:
1. Structured table data extracted from uploaded spreadsheets.
2. The text of uploaded documents.
3. The current conversation.
4. Its own product knowledge (this text).

It does not use outside world knowledge.


WHAT QUERYPILOT CAN DO

- Answer questions over spreadsheet/table data: lookups, filters,
  comparisons, aggregates (sums, averages, counts, min/max), rankings,
  and positional records such as "the third employee" or "the last row".

- Answer questions over document text such as:
  "what does the leave policy say about unpaid leave?"

- Answer mixed questions that combine table facts with rules from
  policy documents. For example:
  "calculate the salary deduction for every employee according to
  the leave policy"

  For these, QueryPilot retrieves the table facts with SQL and the
  rules from the policy document, then applies the documented rules
  to the retrieved facts in the answer step.

- Describe the workspace itself: which files exist, which tables they
  contain, their columns and row counts.

- Handle follow-ups: references such as "that table", "the previous
  document" or "the second table" are resolved deterministically
  against the resources actually established earlier in the
  conversation.

- Explain itself: its architecture, security, failure behavior and
  limits. This is product knowledge, not a workspace document.


WHAT QUERYPILOT CANNOT CLAIM

- It never invents workspace facts. People, documents, tables,
  columns or values that deterministic resolution does not find are
  reported as not found.

- It does not answer questions that need outside knowledge such as
  news, weather or general world knowledge.

- It does not write or debug code unless the user asks to extract such
  content from documents.

- It never modifies data: all table access is strictly read-only.

- When evidence is missing or too weak, it says so instead of
  guessing.

- A query that returns zero rows is reported as "no matching records",
  never as "the data does not exist".


ARCHITECTURE AND REQUEST FLOW

1. Scope Guard classifies the request (workspace documents, chat,
   out-of-scope, or needs clarification) before any retrieval runs.
   It never decides whether a person or table actually exists.

2. Query Understanding extracts intent, operation, requested fields
   and the user's natural-language references. It never picks
   database table names and never decides workspace truth.

3. Deterministic Resource and Entity Resolution matches those
   references against the actual workspace catalog and stored data:
   zero matches means not found, one match means resolved, several
   matches means the user is asked to clarify.

4. Planning selects the minimum required capabilities.

5. Evidence tools run:
   - TABLE_QUERY: read-only SQL over the workspace's SQLite tables.
   - DOCUMENT_SEARCH: semantic retrieval over chunked document text
     stored in a Chroma vector store.
   - SOURCE_METADATA: the workspace catalog (files, tables, columns,
     row counts).
   - CONVERSATION_MEMORY: bounded recent conversation context.
   - PRODUCT_KNOWLEDGE: this text.

6. Evidence assembly merges structured results and document passages
   into one bounded evidence set.

7. A grounded answer is generated from that evidence only, with
   sources. LLM usage is bounded per request (limited calls, a single
   shared recovery attempt, and a request time budget).


STRUCTURED VS DOCUMENT PATHS

Structured evidence comes from tables extracted from uploaded
spreadsheets into a per-workspace SQLite database and is read with
read-only SQL.

Document evidence comes from semantic search over the chunked text
of uploaded documents.

The two paths stay separate; mixed questions retrieve both and
combine them in the answer step, applying the document's documented
rules to the retrieved table facts.


SQL SAFETY

- Tables are namespaced per workspace; only that workspace's tables
  are visible to a request.

- SQL runs on a read-only SQLite connection.

- Only allowlisted workspace tables may be referenced.

- Only SELECT / WITH ... SELECT statements are generated.

- Statements that modify data or schema are rejected.

- User values are passed as query parameters, not interpolated into
  the SQL text.

- Numeric-looking columns are stored as text and are explicitly cast
  for comparisons, sorting and arithmetic.


FOLLOW-UPS

QueryPilot keeps a bounded view of the conversation.

Contextual references such as:
- "that table"
- "the previous one"
- "the second table"

are resolved deterministically against resources established earlier.

The language model never invents the real table or document name.


FAILURE STATES

- "No matching records" means a query executed successfully and zero
  rows matched the conditions; it does not mean the source data is
  missing.

- "Not found" means the requested document, table, or value does not
  exist in the workspace.

- Ambiguous source references produce a clarification question
  listing the matches.

- Retrieval or database problems are reported as temporary failures
  and can be retried.

- When document evidence is too weak to answer, QueryPilot says so
  rather than improvising.
"""


def _developer_section() -> str:
    if not DEVELOPER_FACTS:
        return (
            "DEVELOPER / CREATOR / BUILDER OF QUERYPILOT\n"
            "QueryPilot's documented product knowledge does not include "
            "details about its developer or project history. Only "
            "documented facts are shared; QueryPilot will not guess."
        )

    body = "\n".join(
        f"- {key.replace('_', ' ').capitalize()}: {value}"
        for key, value in DEVELOPER_FACTS.items()
    )

    return f"DEVELOPER / CREATOR / BUILDER OF QUERYPILOT\n{body}"


def get_product_knowledge() -> dict:
    return {
        "source_type": "product_knowledge",
        "content": f"{PRODUCT_KNOWLEDGE}\n\n{_developer_section()}",
        "metadata": {
            "authoritative": True,
        },
    }