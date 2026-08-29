from typing import Any


class TableProcessor:
    """
    Processes table nodes extracted from LlamaParse.

    Responsibilities:

    1. Extract table structure
    2. Normalize headers and rows
    3. Validate table quality
    4. Build retrieval metadata
    5. Build semantic representation
    """

    @staticmethod
    def extract_table(
        table_node: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Convert raw LlamaParse table into
        QueryPilot standard format.

        LlamaParse currently returns:

        {
            "page": 1,
            "rows": [...],
            "csv": "...",
            "html": "...",
            "markdown": "..."
        }

        We treat first row as headers.
        """

        raw_rows = table_node.get(
            "rows",
            [],
        )

        if raw_rows:
            headers = raw_rows[0]
            rows = raw_rows[1:]
        else:
            headers = []
            rows = []

        return {
            "title": f"Table Page {table_node.get('page', 0)}",
            "page": table_node.get(
                "page",
                0,
            ),
            "headers": headers,
            "rows": rows,
        }

    @staticmethod
    def normalize_table(
        headers: list[str],
        rows: list[list[str]],
    ) -> tuple[list[str], list[list[str]]]:
        """
        Clean table structure.

        IMPORTANT: financial tables from LlamaParse almost always have
        a blank header cell over the row-label column, e.g.

            ["", "September 30,<br/>2023", "September 24,<br/>2022"]

        The old logic dropped any column whose HEADER cell was falsy -
        which silently deleted the label column ("Cash and cash
        equivalents", "Total assets", etc.) from every row, since it's
        the column that's almost always unlabeled. That's the column
        that makes the row identifiable at all, so this was destroying
        the table's actual data, not just tidying it.

        Fix: keep every column. A blank header gets a placeholder name
        instead of being used as a signal to delete the column.
        """

        headers = [
            str(header).strip() or f"column_{index}"
            for index, header in enumerate(headers)
        ]

        valid_indexes = list(range(len(headers)))

        normalized_rows = []

        for row in rows:

            cleaned_row = []

            for index in valid_indexes:

                if index < len(row):
                    cleaned_row.append(
                        str(row[index]).strip()
                    )
                else:
                    cleaned_row.append("")

            if not any(cleaned_row):
                continue

            normalized_rows.append(
                cleaned_row
            )

        return headers, normalized_rows

    @staticmethod
    def validate_table(
        headers: list[str],
        rows: list[list[str]],
    ) -> bool:
        """
        Reject obviously broken tables.
        """

        if not headers:
            return False

        if not rows:
            return False

        if len(headers) < 2:
            return False

        return True

    @staticmethod
    def build_metadata(
        document_id: int,
        page: int,
        title: str,
        headers: list[str],
        rows: list[list[str]],
    ) -> dict[str, Any]:
        """
        Metadata stored alongside vectors.
        """

        return {
            "document_id": document_id,
            "page": page,
            "title": title,
            "columns": headers,
            "row_count": len(rows),
        }

    @staticmethod
    def serialize_for_embedding(
        title: str,
        headers: list[str],
        rows: list[list[str]],
    ) -> str:
        """
        Convert table into semantic text
        for embedding storage.
        """

        lines = [
            f"Table: {title}",
            "",
            "Columns:",
            ", ".join(headers),
            "",
            f"Contains {len(rows)} rows.",
            "",
            "Sample Rows:",
        ]

        for row in rows[:10]:

            lines.append(
                " | ".join(
                    str(value)
                    for value in row
                )
            )

        return "\n".join(lines)

    @classmethod
    def process(
        cls,
        table_node: dict[str, Any],
        document_id: int,
    ) -> dict[str, Any] | None:
        """
        Main orchestration method.
        """

        print("\n" + "=" * 80)
        print("RAW TABLE NODE")
        print("=" * 80)
        print(table_node)

        table = cls.extract_table(
            table_node
        )

        print("\nEXTRACTED TABLE")
        print(table)

        headers, rows = cls.normalize_table(
            table["headers"],
            table["rows"],
        )

        print("\nNORMALIZED TABLE")
        print("Headers:")
        print(headers)

        print("\nRow Count:")
        print(len(rows))

        if rows:
            print("\nFirst Row:")
            print(rows[0])

        is_valid = cls.validate_table(
            headers,
            rows,
        )

        print("\nVALIDATION RESULT:")
        print(is_valid)

        if not is_valid:

            print(
                "\nTABLE REJECTED"
            )

            return None

        metadata = cls.build_metadata(
            document_id=document_id,
            page=table["page"],
            title=table["title"],
            headers=headers,
            rows=rows,
        )

        print("\nMETADATA")
        print(metadata)

        semantic_text = (
            cls.serialize_for_embedding(
                title=table["title"],
                headers=headers,
                rows=rows,
            )
        )

        payload = {
            "sqlite_payload": {
                "headers": headers,
                "rows": rows,
            },
            "metadata": metadata,
            "semantic_text": semantic_text,
        }

        print("\nFINAL PAYLOAD CREATED")
        print(
            f"Headers: {len(headers)}"
        )
        print(
            f"Rows: {len(rows)}"
        )

        print("=" * 80)

        return payload