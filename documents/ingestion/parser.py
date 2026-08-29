from pathlib import Path

from django.conf import settings
from llama_cloud import LlamaCloud


class LlamaParseService:
    """
    Thin wrapper around LlamaParse.

    Responsibility:
        File -> LlamaParse -> Raw Parse Result
    """

    client = LlamaCloud(
        api_key=settings.LLAMA_CLOUD_API_KEY,
    )

    @classmethod
    def parse_document(cls, file_path: str):

        uploaded_file = cls.client.files.create(
            file=Path(file_path),
            purpose="parse",
        )

        result = cls.client.parsing.parse(
            file_id=uploaded_file.id,
            tier="agentic",
            version="latest",
            expand=["items"],
        )

        return result


class LlamaAdapter:
    """
    Converts LlamaParse output into QueryPilot primitives.

    Output:
        text_nodes
        table_nodes
    """

    @staticmethod
    def extract_text_nodes(result):

        text_nodes = []

        pages = getattr(
            result.items,
            "pages",
            [],
        )

        for page in pages:

            page_number = getattr(
                page,
                "page_number",
                None,
            )

            for item in page.items:

                item_type = getattr(
                    item,
                    "type",
                    None,
                )

                if item_type == "table":
                    continue

                # IMPORTANT: use "md" (markdown), not "value".
                #
                # LlamaParse emits headings as md = "# Apple Inc." /
                # "## CONSOLIDATED STATEMENTS OF OPERATIONS", but value
                # is just the plain text "Apple Inc." with no marker.
                #
                # StructureAwareChunker's primary heading detector looks
                # for "#"/"##" prefixes (MARKDOWN_HEADING_PATTERN). If we
                # feed it "value", that pattern never matches, headings
                # collapse to a handful of ALL-CAPS fallback matches, and
                # unrelated text gets merged into the wrong section -
                # which is exactly what produced only 4 chunks instead of
                # one per heading.
                #
                # Fall back to "value" only if "md" is missing/empty, so
                # we never drop a node just because it lacks markdown.
                text = getattr(
                    item,
                    "md",
                    None,
                ) or getattr(
                    item,
                    "value",
                    None,
                )

                if not text:
                    continue

                text_nodes.append(
                    {
                        "page": page_number,
                        "text": text,
                    }
                )

        return text_nodes

    @staticmethod
    def extract_table_nodes(result):

        table_nodes = []

        pages = getattr(
            result.items,
            "pages",
            [],
        )

        for page in pages:

            page_number = getattr(
                page,
                "page_number",
                None,
            )

            for item in page.items:

                item_type = getattr(
                    item,
                    "type",
                    None,
                )

                if item_type != "table":
                    continue

                table_nodes.append(
                    {
                        "page": page_number,
                        "rows": getattr(
                            item,
                            "rows",
                            [],
                        ),
                        "csv": getattr(
                            item,
                            "csv",
                            "",
                        ),
                        "html": getattr(
                            item,
                            "html",
                            "",
                        ),
                        "markdown": getattr(
                            item,
                            "md",
                            "",
                        ),
                    }
                )

        return table_nodes