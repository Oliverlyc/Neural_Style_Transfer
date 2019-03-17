import tensorflow as tf
import numpy as np
import scipy
from vgg19_model import VGG19
from PIL import Image
import os
import time
import math

class NeuralStyleInVGG19(object):

    def __init__(self, content_img_path, style_img_path, n_H=480, n_W=640):
        self.content_img_path = content_img_path
        self.style_img_path = style_img_path
        self.n_W = n_W
        self.n_H = n_H
        #内容
        self.content_img = tf.constant(self.preprocess_img(self.content_img_path))
        #风格
        self.style_img = tf.constant(self.preprocess_img(self.style_img_path))
        #生成
        self.generate_img = tf.Variable(self.preprocess_img(self.content_img_path))
        self.create_model()
        self.content_layer = ['block4_conv1']
        self.style_layer = ['block1_conv1',
                            'block2_conv1',
                            'block3_conv1',
                            'block4_conv1',
                            'block5_conv1']
        pass

    #图像预处理
    def preprocess_img(self, img_path):
        img = scipy.misc.imread(img_path, mode='RGB')
        img = scipy.misc.imresize(img, (self.n_H, self.n_W)).astype(np.float32)
        # img[:, :, 0] -= np.mean(img[:, :, 0])
        img[:, :, 0] -= 124
        img[:, :, 1] -= 117
        img[:, :, 2] -= 104
        #img[:, :, 0] /= np.std(img[:, :, 0])
        # img[:, :, 1] -= np.mean(img[:, :, 1])
        #img[:, :, 1] /= np.std(img[:, :, 1])
        # img[:, :, 2] -= np.mean(img[:, :, 2])
        #img[:, :, 2] /= np.std(img[:, :, 2])
        img = np.expand_dims(img, axis=0)
        return img
    def deprocess_img(self, img):
        img[:, :, 0] += 124
        img[:, :, 1] += 117
        img[:, :, 2] += 104
        img = np.clip(img, 0, 255).astype('uint8')
        return img

    def create_model(self):
        input_tensor = tf.concat([self.content_img, self.style_img, self.generate_img], axis=0)
        self.model = VGG19(input_tensor)

    def compute_total_loss(self, content_loss_weight=0.025, style_loss_weight=1):
        loss = 0.0
        print(self.model)
        for layer in self.content_layer:
            loss += content_loss_weight * (self.compute_content_loss(self.model[layer][0, :, :, :], self.model[layer][2, :, :, :]))
        for layer in self.style_layer:
            loss += style_loss_weight * (self.compute_style_loss(self.model[layer][1, :, :, :], self.model[layer][2, :, :, :]))
        tf.summary.scalar('total_loss', loss)
        return loss

    def compute_content_loss(self, content_features, generate_features):
        #内容l2损失
        height, width, channel = [i.value for i in content_features.get_shape()]
        content_size = height * width * channel
        content_loss = tf.nn.l2_loss(generate_features - content_features)/(content_size)
        tf.summary.scalar('content_loss', content_loss)
        return content_loss
        pass

    #风格损失
    def compute_style_loss(self, style_features, generate_features):
        #计算Gram矩阵的偏心协方差矩阵

        height, width, channel = [i.value for i in style_features.get_shape()]
        style_size = height * width * channel
        generate_features = tf.reshape(generate_features, (-1, channel))
        generate_gram = tf.matmul(tf.transpose(generate_features), generate_features)
        style_features = tf.reshape(style_features, (-1, channel))
        style_gram = tf.matmul(tf.transpose(style_features), style_features)
        style_loss = tf.nn.l2_loss(generate_gram - style_gram)/(4 * pow(channel, 2) * pow(width*height, 2))
        tf.summary.scalar('style_loss', style_loss)
        return style_loss


    def generate(self, learning_rate=1.5,epoch=100):
        loss = self.compute_total_loss()
        train_optimize = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        localtime = time.localtime(time.time())
        localtime = str(localtime.tm_mon) +'-'+ str(localtime.tm_mday) + '_' + str(localtime.tm_hour) + '-' + str(localtime.tm_min)
        if not os.path.exists('./generate/' + localtime) :
            os.mkdir('./generate/' + localtime)
        with tf.Session() as sess:
            tf.initialize_all_variables().run()
            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter('./logs', sess.graph)
            for i in range(epoch):
                _, train_loss, generate_image = sess.run([train_optimize, loss, self.generate_img])
                print("[epoch:%d/%d][loss:%.9f]"%(i, epoch, train_loss))
                if(i+1) % 20 == 0:
                    summary_merged = sess.run(merged)
                    writer.add_summary(summary_merged, i+1)
                    generate = self.deprocess_img(generate_image[0])
                    Image.fromarray(generate).save('./generate/' + localtime + '/' + str(i+1) + '.jpg')
                    print("Image saved in ./generate/"+ localtime + '/' + str(i+1) + '.png')
        pass

if __name__ == '__main__':
    content_img_path = './content/2.jpg'
    style_img_path = './style/1.jpg'
    neural_style = NeuralStyleInVGG19(content_img_path, style_img_path)
    neural_style.generate()