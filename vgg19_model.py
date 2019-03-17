import tensorflow as tf
import numpy as np
from scipy.io import loadmat

def load_vgg_mat():
    global vgg_params
    path = './imagenet-vgg-verydeep-19.mat'
    vgg_params = loadmat(path)
    print("Load VGG Mat")
    return vgg_params

#定义卷积层
def conv2D(input_tensor, name, mat_index, strides=[1,1,1,1], padding='SAME'):
    '''

    :param input_tensor: 229x229x3
    :param name:
    :param strides: [1,1,1,1]
    :param padding:
    :return: conv2d
    '''
    with tf.name_scope(name) as scope:
        weights = vgg_params['layers'][0][mat_index][0][0][2][0][0]
        weights = np.transpose(weights, (1, 0, 2, 3))
        tf.summary.histogram(name+'/weights', weights)
        # mat weights W H IN OUT
        bias = vgg_params['layers'][0][mat_index][0][0][2][0][1]
        conv = tf.nn.conv2d(input_tensor, weights, strides, padding=padding, name=scope)
        conv = tf.nn.bias_add(conv, bias.reshape(-1))
        leaky_relu = tf.nn.relu(conv)
        return leaky_relu
    pass
#定义池化层
def pool(input_tensor, name, max_pool=True):
    with tf.name_scope(name) as scope:
        if max_pool is True:
            return tf.nn.max_pool(input_tensor, [1, 3, 3, 1], [1, 2, 2, 1], name=name, padding='SAME')
        else:
            return tf.nn.avg_pool(input_tensor, [1, 3, 3, 1], [1, 2, 2, 1], name=name, padding='SAME')
    pass

def VGG19(input_tensor, BN=None):
    load_vgg_mat()
    #block1
    conv1_1 = conv2D(input_tensor, 'block1_conv1', 0)
    conv1_2 = conv2D(conv1_1, 'block1_conv2', 2)
    pool1 = pool(conv1_2, 'block1_pool', max_pool=True)
    #block2
    conv2_1 = conv2D(pool1, 'block2_conv1', 5)
    conv2_2 = conv2D(conv2_1, 'block2_conv2', 7)
    pool2 = pool(conv2_2, 'block2_pool', max_pool=True)
    #block3
    conv3_1 = conv2D(pool2, 'block3_conv1', 10)
    conv3_2 = conv2D(conv3_1, 'block3_conv2', 12)
    conv3_3 = conv2D(conv3_2, 'block3_conv3', 14)
    conv3_4 = conv2D(conv3_3, 'block3_conv4', 16)
    pool3 = pool(conv3_4, 'block3_pool', max_pool=True)
    #block4
    conv4_1 = conv2D(pool3, 'block4_conv1', 19)
    conv4_2 = conv2D(conv4_1, 'block4_conv2', 21)
    conv4_3 = conv2D(conv4_2, 'block4_conv3', 23)
    conv4_4 = conv2D(conv4_3, 'block4_conv4', 25)
    pool4 = pool(conv4_4, 'block4_pool', max_pool=True)
    #block5
    conv5_1 = conv2D(pool4, 'block5_conv1', 28)
    conv5_2 = conv2D(conv5_1, 'block5_conv2', 30)
    conv5_3 = conv2D(conv5_2, 'block5_conv3', 32)
    conv5_4 = conv2D(conv5_3, 'block5_conv4', 34)
    pool5 = pool(conv5_4, 'block5_pool', max_pool=True)

    features = {'block1_conv1': conv1_1,
               'block2_conv1': conv2_1,
               'block3_conv1': conv3_1,
               'block4_conv1': conv4_1,
               'block5_conv1': conv5_1,
               'block4_conv1': conv5_1}
    return features
    pass

if __name__ == '__main__':
    load_vgg_mat()
    print(vgg_params['meta']['normalization'][0][0][0])