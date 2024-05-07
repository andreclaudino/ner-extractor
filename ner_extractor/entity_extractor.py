from typing import Dict, List, Union
from functools import reduce
from gliner import GLiNER


class EntityExtractor:

    def __init__(self, labels: List[str], pre_trained_model_name: str = "DeepMount00/universal_ner_ita") -> None:
        self.set_labels(labels)
        self._load_entity_extractor_model(pre_trained_model_name)

    def set_labels(self, labels: List[str]) -> None:
        self._labels = labels

    def _load_entity_extractor_model(self, pre_trained_model_name: str):
        self._model = GLiNER.from_pretrained(pre_trained_model_name)
    
    def extract_entities(self, text: str, threshold: float = 0.5) -> List[Dict[str, Union[str, float]]]:
        model_outputs = self._model.predict_entities(text, self._labels, threshold=threshold, multi_label=False, flat_ner=False)
        outputs = []

        for model_output in model_outputs:
            output = dict(
                label = model_output["label"],
                text = model_output["text"],
                score = model_output["score"]
            )

            outputs.append(output)
        
        return outputs
    

