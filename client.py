from docarray import  Document
from jina import Client, DocumentArray


while True:
    da = DocumentArray(Document(text=input('enter prompt>')))
    img_list = input("enter images>").split(' ')
    da.extend([Document(uri=img).load_uri_to_image_tensor() for img in img_list])
    client = Client(port=55288)
    service = input('enter 0 for scores and 1 for rank>')
    if service == '0':
        response = client.post(on='/score', inputs=da)
        print('score:', response[0].text)
    elif service == '1':
        response = client.post(on='/rank', inputs=da)
        print('uri:', response[0].text)
        print('rank:', response[2].text)
        print('score:', response[1].text)