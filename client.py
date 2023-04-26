from docarray import Document
from jina import Client, DocumentArray

img1 = DocumentArray.from_files('./img/*.jpg')
for i in img1:
    i = i.load_uri_to_image_tensor()

img2 = DocumentArray.from_files('./img/*.png')
for i in img2:
    i = i.load_uri_to_image_tensor()

img3 = DocumentArray.from_files('./img/img.png')
for i in img3:
    i = i.load_uri_to_image_tensor()

da = DocumentArray(
    [Document(text='angry', matches=img1),
     Document(text='sea', matches=img2),
     Document(text='sky',matches=img3)]
)
client = Client(port=59181)
responses = client.post(on='/rank', inputs=da)
for response in responses:
    print(response.text)
    for i in response.matches:
        print(f"%2d  %.3f  %s" % (i.scores['rank'].value, i.scores['reward'].value, i.uri))
