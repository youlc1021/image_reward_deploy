from docarray import  Document
from jina import Client, DocumentArray
import numpy as np
import os
import glob

img = DocumentArray.from_files('./img/*.jpg')
for i in img:
    i = i.load_uri_to_image_tensor()

da = DocumentArray(
    [Document(text='This is a photo of a pumpkin',matches=img)]
)
client = Client(port=50847)
response = client.post(on='/rank', inputs=da)
print('rank:', response[0].tensor)
print('rewards:', response[1].tensor)

# c = np.stack((response[0].tensor, response[1].tensor),axis=1)
# print(c)