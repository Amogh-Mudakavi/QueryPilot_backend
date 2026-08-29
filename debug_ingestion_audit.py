from pathlib import Path
import os
import django

os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE",
    "config.settings",
)

django.setup()

from documents.models import Document
from documents.ingestion.vectors import VectorStore
from documents.ingestion.sqlite import SQLiteManager


OUTPUT_FILE = "output.txt"


def write(line=""):
    with open(
        OUTPUT_FILE,
        "a",
        encoding="utf-8",
    ) as f:
        f.write(str(line))
        f.write("\n")


# --------------------------------------------------
# RESET OUTPUT
# --------------------------------------------------

Path(
    OUTPUT_FILE
).write_text(
    "",
    encoding="utf-8",
)

write("=" * 100)
write("QUERYPILOT INGESTION AUDIT")
write("=" * 100)
write()


# --------------------------------------------------
# DOCUMENTS
# --------------------------------------------------

docs = Document.objects.all()

write("DOCUMENTS")
write("-" * 100)

for doc in docs:

    write(
        f"ID={doc.id} | TITLE={doc.title} | WORKSPACE={doc.workspace_id}"
    )

write()
write()


# --------------------------------------------------
# KNOWLEDGE COLLECTION
# --------------------------------------------------

knowledge = (
    VectorStore.knowledge_collection()
)

write("=" * 100)
write("KNOWLEDGE COLLECTION")
write("=" * 100)

write(
    f"COUNT = {knowledge.count()}"
)

all_items = knowledge.get(
    include=[
        "documents",
        "metadatas",
    ]
)

write()

for doc_id, doc_text, meta in zip(
    all_items["ids"],
    all_items["documents"],
    all_items["metadatas"],
):

    write("-" * 100)

    write(f"VECTOR ID: {doc_id}")

    write("METADATA:")
    write(meta)

    write()

    write("DOCUMENT:")
    write(doc_text[:1500])

    write()
    write()


# --------------------------------------------------
# TEXT CHUNKS
# --------------------------------------------------

write("=" * 100)
write("TEXT CHUNKS")
write("=" * 100)

text_only = knowledge.get(
    where={
        "type": "text"
    },
    include=[
        "documents",
        "metadatas",
    ],
)

for doc_id, doc_text, meta in zip(
    text_only["ids"],
    text_only["documents"],
    text_only["metadatas"],
):

    write("-" * 100)

    write(
        f"{doc_id}"
    )

    write(
        f"heading={meta.get('heading')}"
    )

    write(
        f"pages={meta.get('page_start')}->{meta.get('page_end')}"
    )

    write()

    write(doc_text)

    write()
    write()


# --------------------------------------------------
# TABLE METADATA
# --------------------------------------------------

write("=" * 100)
write("TABLE METADATA")
write("=" * 100)

table_only = knowledge.get(
    where={
        "type": "table"
    },
    include=[
        "documents",
        "metadatas",
    ],
)

for doc_id, doc_text, meta in zip(
    table_only["ids"],
    table_only["documents"],
    table_only["metadatas"],
):

    write("-" * 100)

    write(
        f"TABLE VECTOR = {doc_id}"
    )

    write(meta)

    write()

    write(doc_text)

    write()
    write()


# --------------------------------------------------
# SQLITE TABLES
# --------------------------------------------------

write("=" * 100)
write("SQLITE TABLES")
write("=" * 100)

tables = SQLiteManager.list_tables()

for table_name in tables:

    conn = (
        SQLiteManager.get_connection()
    )

    cursor = conn.cursor()

    cursor.execute(
        f'SELECT * FROM "{table_name}" LIMIT 10'
    )

    rows = cursor.fetchall()

    write("-" * 100)

    write(
        f"TABLE NAME: {table_name}"
    )

    write(
        f"ROWS RETURNED: {len(rows)}"
    )

    write()

    for row in rows:

        write(row)

    write()

    conn.close()


# --------------------------------------------------
# DOCUMENT PROFILES
# --------------------------------------------------

write("=" * 100)
write("DOCUMENT PROFILES")
write("=" * 100)

profiles = (
    VectorStore.document_collection()
)

write(
    f"COUNT = {profiles.count()}"
)

results = profiles.get(
    include=[
        "documents",
        "metadatas",
    ]
)

for doc_text, meta in zip(
    results["documents"],
    results["metadatas"],
):

    write("-" * 100)

    write(meta)

    write()

    write(doc_text)

    write()
    write()


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

write("=" * 100)
write("SUMMARY")
write("=" * 100)

write(
    f"Knowledge Vectors: {knowledge.count()}"
)

write(
    f"Document Profiles: {profiles.count()}"
)

write(
    f"SQLite Tables: {len(tables)}"
)

write()
write(
    "AUDIT COMPLETE"
)

print(
    f"\nSaved audit to {OUTPUT_FILE}"
)