import os.path

import ImageReward as RM
import torch
from jina import Executor, requests, DocumentArray
from docarray import Document
from PIL import Image
import numpy as np


class TextReward(Executor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = RM.load("ImageReward-v1.0")

    @requests(on='/rank')
    def rank(self, docs: DocumentArray, **kwargs):
        # for d in docs
        ranking, rewards = self._rank(docs)
        ranking = np.array(ranking)
        rewards = np.array(rewards)
        return DocumentArray([Document(tensor=ranking),Document(tensor=rewards)])

    def _rank(self, docs:DocumentArray):
        rnk, rwd = [], []
        img = []
        with torch.no_grad():
            for i in docs['@m']:
                i = Image.fromarray(i.tensor)
                img.append(i)
            print(docs['@r'][0].text)
            ranking, rewards = self.model.inference_rank(docs['@r'][0].text, img)
            rnk.append(ranking)
            rwd.append(rewards)
        return ranking, rewards
