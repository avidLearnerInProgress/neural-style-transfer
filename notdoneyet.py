import time, cv2, os, sys
from PIL import Image as im
import numpy as np

from keras import backend as K
from keras.models import Model
from keras.applications.vgg16 import VGG16
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave


def gram_matrix(a):
	features = K.batch_flatten(K.permute_dimensions(a, (2, 0, 1)))
	gram = K.dot(features, K.transpose(features))
	return gram

def content_loss(content, combination):
	res = K.sum(K.square(content - combination))
	return res

def style_loss(style, combination, height, width):
	s = gram_matrix(style)
	c = gram_matrix(combination)
	channels = 3
	size = height * width
	return K.sum(K.square(s-c)) / (4. * (channels ** 2) * (size ** 2))

def total_variation_loss(a, height, width):
	m = K.square(a[:, :height - 1, :width - 1, :] - a[:, 1:, width - 1, :])
	n = K.square(a[:, :height - 1, :width - 1, :] - a[:, :height - 1, 1:, :])
	return K.sum(K.pow(m + n, 1.25))



def main():
	height = 512
	width = 512

	cimg_path = 'images/content/7.jpg'
	cimg = im.open(cimg_path)
	cimg = cimg.resize((width, height))
	cv2.imshow('content', np.asarray(cimg))
	cv2.waitKey(0)

	simg_path = 'images/style/7.jpg'
	simg = im.open(simg_path)
	simg = simg.resize((width, height))
	cv2.imshow('style', np.asarray(simg))
	cv2.waitKey(0)
	
	carr = np.asarray(cimg, dtype = 'float32')
	carr = np.expand_dims(carr, axis = 0)

	sarr = np.asarray(simg, dtype = 'float32')
	sarr = np.expand_dims(sarr, axis = 0)

	print(carr.shape)
	print(sarr.shape)

	carr[:, :, :, 0] -= 103.939
	carr[:, :, :, 1] -= 116.779
	carr[:, :, :, 2] -= 123.68
	carr = carr[:, :, :, ::-1]

	sarr[:, :, :, 0] -= 103.939
	sarr[:, :, :, 1] -= 116.779
	sarr[:, :, :, 2] -= 123.68
	sarr = sarr[:, :, :, ::-1]

	cimg = K.variable(carr)
	simg = K.variable(sarr)
	csimg = K.placeholder((1, height, width, 3))
	iptensor = K.concatenate([cimg, simg, csimg], axis = 0)

	model = VGG16(input_tensor = iptensor, weights = 'imagenet', include_top = False)
	layers = dict([(layer.name, layer.output) for layer in model.layers])
	print(layers)
	
	cweight = 0.025
	sweight = 5.0
	vweight = 1.0

	loss = K.variable(0.)

	#Computing content loss
	layer_features = layers['block2_conv2']
	cimg_features = layer_features[0, :, :, :]
	csimg_features = layer_features[2, :, :, :]
	loss += cweight * content_loss(cimg_features, csimg_features)

	#Computing style loss
	feature_layers = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3']
	for lname in feature_layers:
		lfeatures = layers[lname]
		sfeatures = lfeatures[1, :, :, :]
		csfeatures = lfeatures[2, :, :, :]
		sl = style_loss(sfeatures, csimg_features, height, width)
		loss += (sweight / len(feature_layers)) * sl

	#Computing total variation loss
	loss += vweight * total_variation_loss(csimg, height, width)

	#Solving optimisation problem
	gradients = K.gradients(loss, csimg)
	op = [loss]
	op += gradients
	fop = K.function([csimg], op)


	def eval_loss_grads(a):
		a = a.reshape([1, height, width, 3])
		out = fop([a])
		loss_val = out[0]
		grad_vals = out[1].flatten().astype('float64')
		return loss_val, grad_vals

	class Evaluator(object):
		def __init__(self):
			self.loss_val = None
			self.grad_vals = None

		def loss(self, a):
			assert self.loss_val is None
			loss_val, grad_vals = eval_loss_grads(a)
			self.loss_val = loss_val
			self.grad_vals = grad_vals
			return self.loss_val

		def grads(self, a):
			assert self.loss_val is not None
			grad_vals = np.copy(self.grad_vals)
			self.loss_val = None
			self.grad_vals = None
			return grad_vals


	evaluator = Evaluator()
	a = np.random.uniform(0, 255, (1, height, width, 3)) - 128
	itr = 15
	for i in range(itr):
		print("Iteration: ", i)
		start_t = time.time()
		a, min_val, inf = fmin_l_bfgs_b(evaluator.loss, a.flatten(), fprime = evaluator.grads, maxfun = 20)
		print("Closs: ", min_val)
		end_t = time.time()
		print("Iteration %d completed in %d" % (i, end_t - start_t))

	a = a.reshape((height, width, 3))
	a = a[:, :, ::-1]
	a[:, :, 0] += 103.939
	a[:, :, 1] += 116.779
	a[:, :, 2] += 123.68
	a = np.clip(a, 0, 255).astype('uint8')
	cv2.imshow('styletransfer', np.asarray(a))
	cv2.waitKey(0)

if __name__ == '__main__':
	main()

