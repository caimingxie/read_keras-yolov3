from keras import backend as K
import tensorflow as tf

from keras.layers import  Input

'''
grid_shape =(13,13)

grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
        [1, grid_shape[1], 1, 1])


grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
        [grid_shape[0], 1, 1, 1])

grid = K.concatenate([grid_x, grid_y])


with tf.Session() as sess:
    print (sess.run(grid))



image_shape = [512.0,960.0]
input_shape=[16.0,30.0]

image_shape=tf.convert_to_tensor(image_shape)
input_shape=tf.convert_to_tensor(input_shape)
new_shape = K.round(image_shape * K.min(input_shape/image_shape))
offset = (input_shape-new_shape)/2./input_shape

with tf.Session() as sess:
    print(sess.run(new_shape))
    print(sess.run(offset))


y_true = [Input(shape=(416//{0:32, 1:16, 2:8}[l], 416//{0:32, 1:16, 2:8}[l], \
        9//3, 80+5)) for l in range(3)]

print(y_true)


image_input = Input(shape=(None, None, 3))

print(image_input)
'''
import numpy as np
input_shape=np.array([416,416])
grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(3)]
print(grid_shapes)