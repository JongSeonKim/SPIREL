import numpy as np
import tensorflow as tf

from params import Params
from icf import ICF

def main(_):
	params = Params()
	dataset = np.load('../data/' + params.data_path + '.npy', allow_pickle=True)
	[user_train, user_test, user_num, item_num] = dataset
	
	print("Loading", params.data_path, '.npy...')
	print("# users : ", user_num)
	print("# items : ", item_num)
	
	config = tf.compat.v1.ConfigProto()
	config.gpu_options.allow_growth = True
	
	with tf.Graph().as_default():
		sess = tf.compat.v1.Session(config=config)
		with sess.as_default():
			model = ICF(dataset, params, sess)
			model.process()
			
if __name__ == "__main__":
	tf.compat.v1.app.run()