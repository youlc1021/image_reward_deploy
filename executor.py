import ImageReward as RM
import torch
from jina import Executor, requests, DocumentArray
from docarray import Document
from PIL import Image


class TextReward(Executor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = RM.load("ImageReward-v1.0")

    @requests(on='/rank')
    def rank(self, docs: DocumentArray, **kwargs):
        for doc in docs:
            self._rank(doc)
            doc.matches = sorted(doc.matches, key=lambda _s:_s.scores['rank'].value)

    def _rank(self, doc: Document):
        img = []
        for i in doc.matches:
            img.append(Image.fromarray(i.tensor))
        with torch.no_grad():
            ranking, rewards = self.model.inference_rank(doc.text, img)
        if len(doc.matches) == 1:
            doc.matches[0].scores['rank'].value, doc.matches[0].scores['reward'].value = ranking, rewards
        else:
            j = 0
            for i in doc.matches:
                i.scores['rank'].value, i.scores['reward'].value = ranking[j], rewards[j]
                j += 1
