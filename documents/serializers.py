"""Serializers module for QueryPilot."""

import csv
import hashlib
import io
import zipfile

import fitz
from rest_framework import serializers

from workspace.models import Workspace

from .models import Document


class DocumentUploadSerializer(serializers.Serializer):
    workspace_id = serializers.IntegerField()
    file = serializers.FileField()

    MAX_FILE_SIZE = 50 * 1024 * 1024

    ALLOWED_EXTENSIONS = {
        ".pdf",
        ".docx",
        ".xlsx",
        ".csv",
    }

    MAX_ARCHIVE_UNCOMPRESSED_SIZE = 250 * 1024 * 1024
    MAX_ARCHIVE_MEMBERS = 10000
    MAX_COMPRESSION_RATIO = 1000
    CSV_SAMPLE_SIZE = 1024 * 1024

    def validate_workspace_id(self, value):

        request = self.context["request"]

        if not Workspace.objects.filter(
            id=value,
            owner=request.user,
        ).exists():
            raise serializers.ValidationError("Workspace not found.")

        return value

    def validate_file(self, file):

        if file.size > self.MAX_FILE_SIZE:
            raise serializers.ValidationError("File size exceeds 50 MB limit.")

        filename = file.name.lower()

        extension = None

        for allowed in self.ALLOWED_EXTENSIONS:
            if filename.endswith(allowed):
                extension = allowed
                break

        if extension is None:
            raise serializers.ValidationError("Unsupported file type.")

        if extension == ".pdf":
            self._validate_pdf(file)

        elif extension == ".xlsx":
            self._validate_xlsx(file)

        elif extension == ".docx":
            self._validate_docx(file)

        elif extension == ".csv":
            self._validate_csv(file)

        file.seek(0)

        return file

    @staticmethod
    def calculate_file_hash(file) -> str:

        file.seek(0)

        digest = hashlib.sha256()

        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

        file.seek(0)

        return digest.hexdigest()

    def _validate_pdf(self, file):

        file.seek(0)

        if file.read(5) != b"%PDF-":
            raise serializers.ValidationError("Invalid PDF file.")

        file.seek(0)

        try:
            temporary_path = getattr(
                file,
                "temporary_file_path",
                None,
            )

            if callable(temporary_path):
                pdf = fitz.open(filename=temporary_path())

            else:
                pdf = fitz.open(
                    stream=file.read(),
                    filetype="pdf",
                )

            try:
                if pdf.page_count == 0:
                    raise serializers.ValidationError("PDF contains no pages.")

            finally:
                pdf.close()

        except serializers.ValidationError:
            raise

        except Exception as exc:
            raise serializers.ValidationError("Corrupted PDF file.") from exc

        finally:
            file.seek(0)

    def _validate_xlsx(self, file):

        self._validate_zip(
            file,
            required_member="xl/workbook.xml",
        )

    def _validate_docx(self, file):

        self._validate_zip(
            file,
            required_member="word/document.xml",
        )

    def _validate_zip(
        self,
        file,
        required_member,
    ):

        file.seek(0)

        if file.read(2) != b"PK":
            raise serializers.ValidationError("Invalid document archive.")

        file.seek(0)

        try:
            with zipfile.ZipFile(file) as archive:
                members = archive.infolist()

                if len(members) > self.MAX_ARCHIVE_MEMBERS:
                    raise serializers.ValidationError(
                        "Document contains too many archive entries."
                    )

                total_size = 0

                for member in members:
                    name = member.filename.replace("\\", "/")

                    if name.startswith("/") or "../" in name or name == "..":
                        raise serializers.ValidationError("Invalid document archive.")

                    if (
                        "vbaProject.bin" in name
                        or "/oleObject" in name
                        or name.startswith("embeddings/")
                    ):
                        raise serializers.ValidationError(
                            "Document contains unsupported embedded content."
                        )

                    total_size += member.file_size

                    if total_size > self.MAX_ARCHIVE_UNCOMPRESSED_SIZE:
                        raise serializers.ValidationError(
                            "Document expands beyond the allowed limit."
                        )

                    compressed_size = member.compress_size

                    if compressed_size > 0:
                        ratio = member.file_size / compressed_size

                        if ratio > self.MAX_COMPRESSION_RATIO:
                            raise serializers.ValidationError(
                                "Document contains suspicious compression."
                            )

                if required_member not in archive.namelist():
                    raise serializers.ValidationError("Invalid document structure.")

        except serializers.ValidationError:
            raise

        except zipfile.BadZipFile as exc:
            raise serializers.ValidationError("Corrupted document file.") from exc

        except Exception as exc:
            raise serializers.ValidationError("Invalid document file.") from exc

        finally:
            file.seek(0)

    def _validate_csv(self, file):

        try:
            file.seek(0)

            sample = file.read(self.CSV_SAMPLE_SIZE)

            if not sample:
                raise serializers.ValidationError("CSV contains no data.")

            content = sample.decode(
                "utf-8",
                errors="strict",
            )

            reader = csv.reader(io.StringIO(content))

            for _ in range(5):
                next(reader)

        except serializers.ValidationError:
            raise

        except UnicodeDecodeError as exc:
            raise serializers.ValidationError("CSV must be UTF-8 encoded.") from exc

        except StopIteration:
            if not content.strip():
                raise serializers.ValidationError("CSV contains no data.")

        except Exception as exc:
            raise serializers.ValidationError("Invalid CSV file.") from exc

        finally:
            file.seek(0)


class DocumentListSerializer(serializers.ModelSerializer):
    class Meta:
        model = Document

        fields = [
            "id",
            "title",
            "source_type",
            "status",
            "uploaded_at",
        ]
