import requests, zipfile, io, os


os.mkdir('../data')
url = 'http://emodb.bilderbar.info/download/download.zip'

r = requests.get(url)
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall("../data")

