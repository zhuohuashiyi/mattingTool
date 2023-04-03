#!/usr/bin/env python
from __future__ import print_function
import os
from os.path import splitext, join
import numpy as np
from scipy import misc
from keras import backend as K
from keras.models import model_from_json, load_model
import tensorflow as tf
import layers_builder as layers
from glob import glob
from utils import utils
from keras.utils.generic_utils import CustomObjectScope
import cv2
import math
import matplotlib.pyplot as plt


from imageio import imread
DATA_MEAN = np.array([[[123.68, 116.779, 103.939]]])  # RGB order


class PSPNet(object):  # pspnet的基函数，供继承使用
    def __init__(self, nb_classes, resnet_layers, input_shape, weights):
        self.input_shape = input_shape  # 输入数据的大小
        self.num_classes = nb_classes   # 元素类别的数量

        json_path = join("weights", "keras", weights + ".json")  # 网络参数文件的路径
        h5_path = join("weights", "keras", weights + ".h5")
        # 以下开始处理网络参数文件
        if 'pspnet' in weights:
            if os.path.isfile(json_path) and os.path.isfile(h5_path):
                print("Keras model & weights found, loading...")
                with CustomObjectScope({'Interp': layers.Interp}):
                    with open(json_path) as file_handle:
                        self.model = model_from_json(file_handle.read())
                self.model.load_weights(h5_path)
            else:
                print("No Keras model & weights found, import from npy weights.")
                self.model = layers.build_pspnet(nb_classes=nb_classes,
                                                 resnet_layers=resnet_layers,
                                                 input_shape=self.input_shape)
                self.set_npy_weights(weights)
        else:
            print('Load pre-trained weights')
            self.model = load_model(weights)

    def predict(self, img, flip_evaluation=False):
        # 将一张符合输入形状要求的图片转化成一个语义分割的结果图片
        if img.shape[0:2] != self.input_shape: # 不符合大小要求需要放缩
            print(
                "Input %s not fitting for network size %s, resizing. You may want to try sliding prediction for better results." % (
                img.shape[0:2], self.input_shape))
            img = misc.imresize(img, self.input_shape)
        # 输入数据的预处理
        img = img - DATA_MEAN
        img = img[:, :, ::-1]  
        img = img.astype('float32')
        probs = self.feed_forward(img, flip_evaluation) # 调用前向计算函数进行计算
        return probs

    def predict_sliding(self, full_img, flip_evaluation):
        """
        Predict on tiles of exactly the network input shape.
        This way nothing gets squeezed.
        """
        tile_size = self.input_shape
        classes = self.num_classes
        overlap = 1 / 3

        stride = math.ceil(tile_size[0] * (1 - overlap))
        tile_rows = max(int(math.ceil((full_img.shape[0] - tile_size[0]) / stride) + 1), 1)  # strided convolution formula
        tile_cols = max(int(math.ceil((full_img.shape[1] - tile_size[1]) / stride) + 1), 1)
        print("Need %i x %i prediction tiles @ stride %i px" % (tile_cols, tile_rows, stride))
        full_probs = np.zeros((full_img.shape[0], full_img.shape[1], classes))
        count_predictions = np.zeros((full_img.shape[0], full_img.shape[1], classes))
        tile_counter = 0
        for row in range(tile_rows):
            for col in range(tile_cols):
                x1 = int(col * stride)
                y1 = int(row * stride)
                x2 = min(x1 + tile_size[1], full_img.shape[1])
                y2 = min(y1 + tile_size[0], full_img.shape[0])
                x1 = max(int(x2 - tile_size[1]), 0)  # for portrait images the x1 underflows sometimes
                y1 = max(int(y2 - tile_size[0]), 0)  # for very few rows y1 underflows

                img = full_img[y1:y2, x1:x2]
                padded_img = self.pad_image(img, tile_size)
                plt.imshow(padded_img)
                plt.show()
                tile_counter += 1
                print("Predicting tile %i" % tile_counter)
                padded_prediction = self.predict(padded_img, flip_evaluation)
                prediction = padded_prediction[0:img.shape[0], 0:img.shape[1], :]
                count_predictions[y1:y2, x1:x2] += 1
                full_probs[y1:y2, x1:x2] += prediction  # accumulate the predictions also in the overlapping regions

        # average the predictions in the overlapping regions
        full_probs /= count_predictions
        # visualize normalization Weights
        # plt.imshow(np.mean(count_predictions, axis=2))
        # plt.show()
        return full_probs

    @staticmethod
    def pad_image(img, target_size):
        """Pad an image up to the target size."""
        rows_missing = target_size[0] - img.shape[0]
        cols_missing = target_size[1] - img.shape[1]
        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, 0)), 'constant')
        return padded_img

    def predict_multi_scale(self, img, flip_evaluation, sliding_evaluation, scales):
       # 以不同的尺度对图片进行预测

        full_probs = np.zeros((img.shape[0], img.shape[1], self.num_classes))
        h_ori, w_ori = img.shape[:2]

        print("Started prediction...")
        for scale in scales:
            print("Predicting image scaled by %f" % scale)
            scaled_img = misc.imresize(img, size=scale, interp="bilinear") # 根据指定比例放缩图片

            if sliding_evaluation:
                scaled_probs = self.predict_sliding(scaled_img, flip_evaluation)
            else:
                scaled_probs = self.predict(scaled_img, flip_evaluation)  # 这里调用predict函数

            # scale probs up to full size
            # visualize_prediction(probs)
            probs = cv2.resize(scaled_probs, (w_ori, h_ori)) # 放缩为原来的大小
            full_probs += probs
        full_probs /= len(scales)
        print("Finished prediction...")

        return full_probs

    def feed_forward(self, data, flip_evaluation=False): # 神经网络的前向计算函数
        assert data.shape == (self.input_shape[0], self.input_shape[1], 3)

        if flip_evaluation:
            print("Predict flipped")
            input_with_flipped = np.array(
                [data, np.flip(data, axis=1)])
            prediction_with_flipped = self.model.predict(input_with_flipped)
            prediction = (prediction_with_flipped[
                          0] + np.fliplr(prediction_with_flipped[1])) / 2.0
        else:
            prediction = self.model.predict(np.expand_dims(data, 0))[0]
        return prediction

    def set_npy_weights(self, weights_path):
        npy_weights_path = join("weights", "npy", weights_path + ".npy")
        json_path = join("weights", "keras", weights_path + ".json")
        h5_path = join("weights", "keras", weights_path + ".h5")

        print("Importing weights from %s" % npy_weights_path)
        weights = np.load(npy_weights_path, encoding='bytes').item()
        for layer in self.model.layers:
            print(layer.name)
            if layer.name[:4] == 'conv' and layer.name[-2:] == 'bn':
                mean = weights[layer.name.encode()][
                    'mean'.encode()].reshape(-1)
                variance = weights[layer.name.encode()][
                    'variance'.encode()].reshape(-1)
                scale = weights[layer.name.encode()][
                    'scale'.encode()].reshape(-1)
                offset = weights[layer.name.encode()][
                    'offset'.encode()].reshape(-1)

                self.model.get_layer(layer.name).set_weights(
                    [scale, offset, mean, variance])

            elif layer.name[:4] == 'conv' and not layer.name[-4:] == 'relu':
                try:
                    weight = weights[layer.name.encode()]['weights'.encode()]
                    self.model.get_layer(layer.name).set_weights([weight])
                except Exception as err:
                    biases = weights[layer.name.encode()]['biases'.encode()]
                    self.model.get_layer(layer.name).set_weights([weight,
                                                                  biases])
        print('Finished importing weights.')

        print("Writing keras model & weights")
        json_string = self.model.to_json()
        with open(json_path, 'w') as file_handle:
            file_handle.write(json_string)
        self.model.save_weights(h5_path)
        print("Finished writing Keras model & weights")


class PSPNet50(PSPNet): # 基于50-layer的ResNet建立的pspnet
   

    def __init__(self, nb_classes, weights, input_shape):
        PSPNet.__init__(self, nb_classes=nb_classes, resnet_layers=50,
                        input_shape=input_shape, weights=weights)


class PSPNet101(PSPNet): # 基于101-layer的ResNet建立的pspnet

    def __init__(self, nb_classes, weights, input_shape):
        PSPNet.__init__(self, nb_classes=nb_classes, resnet_layers=101,
                        input_shape=input_shape, weights=weights)


def predict(input_path, output_path, model):  # 具体的语义分割操作函数，供外界调用
    images = glob(input_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    sess = tf.Session()
    K.set_session(sess)
    with sess.as_default():
        if "pspnet50" in model:   # 以下根据参数选择建立的模型
            pspnet = PSPNet50(nb_classes=150, input_shape=(473, 473),
                              weights=model)
        elif "pspnet101" in model:
            if "cityscapes" in model:
                pspnet = PSPNet101(nb_classes=19, input_shape=(713, 713),
                                   weights=model)
            if "voc2012" in model:
                pspnet = PSPNet101(nb_classes=21, input_shape=(473, 473),
                                   weights=model)
        else:
            print("Network architecture not implemented.")
        EVALUATION_SCALES = [1.0]
        for i, img_path in enumerate(images):  # 开始处理图片
            print("Processing image {} / {}".format(i + 1, len(images)))
            img = imread(img_path, pilmode='RGB')
            probs = pspnet.predict_multi_scale(img, True, False, EVALUATION_SCALES) # 处理图片的具体函数
            cm = np.argmax(probs, axis=2)
            pm = np.max(probs, axis=2)
            colored_class_image = utils.color_class_image(cm, model)
            alpha_blended = 0.5 * colored_class_image + 0.5 * img
            filename, ext = splitext(output_path)  # 保存结果图片
            misc.imsave(filename + "_seg_read" + ext, cm)
            misc.imsave(filename + "_seg" + ext, colored_class_image)
            misc.imsave(filename + "_probs" + ext, pm)
            misc.imsave(filename + "_seg_blended" + ext, alpha_blended)


