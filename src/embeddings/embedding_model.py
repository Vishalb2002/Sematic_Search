from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """
    Singleton class for loading the SBERT model.
    """

    _model = None

    @classmethod
    def get_model(
        cls,
        model_name: str = "all-MiniLM-L6-v2"
    ):

        if cls._model is None:

            print("Loading SBERT Model...")

            cls._model = SentenceTransformer(model_name)

            print("SBERT Loaded Successfully.\n")

        return cls._model