import os.path

import ImageReward as RM
import torch
from jina import Executor, requests, DocumentArray
from docarray import Document


class TextReward(Executor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = RM.load("ImageReward-v1.0")
        self.model_score = self.model.score
        self.model_rank = self.model.inference_rank

    def save_images(self, img_list):
        for i in img_list:
            i.save_image_tensor_to_file('./saved/' + i.uri.split('/')[-1])

    @requests(on='/score')
    def score(self, docs: DocumentArray, **kwargs):
        # can't access through client local uri
        # save images to server
        self.save_images(docs[1:])
        # uri from server
        img_list = []
        for i in docs[1:]:
            img_list.append('./saved/'+ i.uri.split('/')[-1])
        # input text and uri to model and get a list
        with torch.no_grad():
            scores = self.model_score(docs[0].text, img_list)
        # return result as da
        result = DocumentArray(Document(text=str(scores)))
        # delete saved images
        for i in docs[1:]:
            os.remove('./saved/' + i.uri.split('/')[-1])
        return result

    @requests(on='/rank')
    def rank(self, docs: DocumentArray, **kwargs):
        self.save_images(docs[1:])
        img_list = []
        for i in docs[1:]:
            img_list.append('./saved/' + i.uri.split('/')[-1])
        ranking, rewards = self._rank(docs[0].text, img_list)
        result = DocumentArray(Document(text=str(docs[1:,'uri'])))
        result.extend([Document(text=str(rewards)),Document(text=str(ranking))])
        for i in docs[1:]:
            os.remove('./saved/' + i.uri.split('/')[-1])
        return result

    def _rank(self, prompt, img_list):
        with torch.no_grad():
            ranking, rewards = self.model_rank(prompt, img_list)
        return ranking, rewards
