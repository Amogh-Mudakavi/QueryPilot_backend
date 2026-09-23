from documents.ingestion.pipeline import (
    IngestionPipeline,
)


class IngestionService:

    @staticmethod
    def process(document):

        return IngestionPipeline.run(
            document
        )