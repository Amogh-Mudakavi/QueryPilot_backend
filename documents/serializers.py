import csv
import hashlib
import io
import zipfile

import fitz

from rest_framework import serializers

from workspace.models import Workspace
from .models import Document


class DocumentUploadSerializer(serializers.Serializer):
    """
    Validates an uploaded document before ingestion.
    """

    workspace_id = serializers.IntegerField()
    conversation_id = serializers.IntegerField(required=True)
    file = serializers.FileField()

    MAX_FILE_SIZE = 50 * 1024 * 1024

    ALLOWED_EXTENSIONS = {
        ".pdf",
        ".docx",
        ".xlsx",
        ".csv",
    }

    def validate_workspace_id(self, value):
        request = self.context["request"]

        if not Workspace.objects.filter(
            id=value,
            owner=request.user,
        ).exists():
            raise serializers.ValidationError(
                "Workspace not found."
            )

        return value

    def validate_file(self, file):
        if file.size > self.MAX_FILE_SIZE:
            raise serializers.ValidationError(
                "File size exceeds 50 MB limit."
            )

        filename = file.name.lower()

        extension = None

        for allowed in self.ALLOWED_EXTENSIONS:
            if filename.endswith(allowed):
                extension = allowed
                break

        if extension is None:
            raise serializers.ValidationError(
                "Unsupported file type."
            )

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
        if file.read(5) != b"%PDF-":
            raise serializers.ValidationError(
                "Invalid PDF file."
            )

        file.seek(0)

        try:
            pdf = fitz.open(
                stream=file.read(),
                filetype="pdf",
            )

            if pdf.page_count == 0:
                raise serializers.ValidationError(
                    "PDF contains no pages."
                )

            pdf.close()

        except serializers.ValidationError:
            raise

        except Exception as exc:
            raise serializers.ValidationError(
                "Corrupted PDF file."
            ) from exc

        finally:
            file.seek(0)

    def _validate_xlsx(self, file):
        if file.read(2) != b"PK":
            raise serializers.ValidationError(
                "Invalid XLSX file."
            )

        file.seek(0)

        try:
            with zipfile.ZipFile(file) as archive:
                if "xl/workbook.xml" not in archive.namelist():
                    raise serializers.ValidationError(
                        "Invalid XLSX structure."
                    )

        except serializers.ValidationError:
            raise

        except zipfile.BadZipFile as exc:
            raise serializers.ValidationError(
                "Corrupted XLSX file."
            ) from exc

        finally:
            file.seek(0)

    def _validate_docx(self, file):
        if file.read(2) != b"PK":
            raise serializers.ValidationError(
                "Invalid DOCX file."
            )

        file.seek(0)

        try:
            with zipfile.ZipFile(file) as archive:
                if "word/document.xml" not in archive.namelist():
                    raise serializers.ValidationError(
                        "Invalid DOCX structure."
                    )

        except serializers.ValidationError:
            raise

        except zipfile.BadZipFile as exc:
            raise serializers.ValidationError(
                "Corrupted DOCX file."
            ) from exc

        finally:
            file.seek(0)

    def _validate_csv(self, file):
        try:
            file.seek(0)

            content = file.read().decode(
                "utf-8",
                errors="strict",
            )

            reader = csv.reader(
                io.StringIO(content)
            )

            for _ in range(5):
                next(reader)

        except UnicodeDecodeError as exc:
            raise serializers.ValidationError(
                "CSV must be UTF-8 encoded."
            ) from exc

        except StopIteration:
            if not content.strip():
                raise serializers.ValidationError(
                    "CSV contains no data."
                )

        except Exception as exc:
            raise serializers.ValidationError(
                "Invalid CSV file."
            ) from exc

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