from sentence_transformers import CrossEncoder


class CrossEncoderModel:
    """
    Singleton class for loading the CrossEncoder model.
    """

    _model = None

    @classmethod
    def get_model(
        cls,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):

        if cls._model is None:

            # print("Loading CrossEncoder...")

            cls._model = CrossEncoder(model_name)

            print("CrossEncoder Loaded.\n")

        return cls._model