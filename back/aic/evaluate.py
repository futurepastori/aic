import os
import cv2
import json
import numpy as np

from PIL import Image
from torchvision.transforms import transforms
from aic.generate import generate

path = os.path.join(os.getcwd(), 'data')

transform = transforms.Compose([ 
	transforms.Resize(256),
	transforms.RandomCrop(224),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize((0.485, 0.456, 0.406),
						 (0.229, 0.224, 0.225))])

def evaluate(src):
	path = os.path.join('.', 'data')
	image = Image.open(src)
	image_transform = transform(image)

	print(os.getcwd())
	with open(os.path.join(path, 'json', 'wordmap.json')) as j:
		word_map = json.load(j)

	return generate(image, image_transform, word_map)

