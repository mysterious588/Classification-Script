import argparse
import models

parser = argparse.ArgumentParser()

# image path
parser.add_argument('image_path', action="store")

# checkpoint
parser.add_argument('model', action="store")

parser.add_argument('--gpu', action="store_true", help= 'add to train on gpu', default=False,  dest='train_on_gpu')
parser.add_argument('--topk', action="store", help= 'prints top k number of classes', default=5,  dest='topk', type= int)
parser.add_argument('--category_names', action="store", help= 'json file of the category names', default=None,  dest='category_names')

results = parser.parse_args()

image_path = results.image_path
model = results.model
train_on_gpu = results.train_on_gpu
category_names = results.category_names
topk = results.topk


models.predict(image_path, model, train_on_gpu = train_on_gpu, category_names = category_names, topk = topk)
