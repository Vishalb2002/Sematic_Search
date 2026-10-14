
import spacy


class NERExtractor:
    """
    Baseline Named Entity Recognition using spaCy.
    Detects entity mentions and their predicted entity types.
    """

    def __init__(self, model_name="en_core_web_sm"):
        self.nlp = spacy.load(model_name)

    def extract(self, text):
        if not isinstance(text, str) or not text.strip():
            return []

        doc = self.nlp(text)
        entities = []

        for ent in doc.ents:
            entities.append({
                "mention": ent.text,
                "entity_type": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "context": self._get_context(doc, ent),
            })

        return entities

    @staticmethod
    def _get_context(doc, entity):
        """Return the sentence containing the entity."""
        for sentence in doc.sents:
            if (
                sentence.start_char <= entity.start_char
                and sentence.end_char >= entity.end_char
            ):
                return sentence.text

        return entity.sent.text


if __name__ == "__main__":
    extractor = NERExtractor()

    text = (
        "Google developed a new AI model. "
        "Geoffrey Hinton contributed to neural network research."
    )

    results = extractor.extract(text)

    for entity in results:
        print(entity)
