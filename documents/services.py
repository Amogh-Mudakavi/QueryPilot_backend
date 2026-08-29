from documents.ingestion.pipeline import run_pipeline


class IngestionService:

    @staticmethod
    def process(document):

        run_pipeline(document)