# encoding: UTF-8

from tombstone.core.metric import ConfusionMatrix
from PIL import Image
import numpy as np
import os
import caffe


def read_image(path):
    image = Image.open(path)
    image = np.array(image, dtype=np.float32)
    image = image[:, :, ::-1]
    image -= np.array((104.00698793, 116.66876762, 122.67891434))
    image = image.transpose((2, 0, 1))
    return image


def read_label(path):
    label = Image.open(path)
    label = np.array(label, dtype=np.float32)
    return label[:, :, 0]


def reader(root):
    jpeg = os.path.join(root, 'JPEGImages')
    seg = os.path.join(root, 'SegmentationClass')
    with open('data/pascal/seg11valid.txt') as val_set:
        for i, serial in enumerate(val_set, start=1):
            serial = serial.strip()
            if not serial:
                break
            image = os.path.join(jpeg, '{}.jpg'.format(serial))
            label = os.path.join(seg, '{}.png'.format(serial))
            yield i, serial, read_image(image), read_label(label)


def main():
    root = 'data/pascal/VOCdevkit/VOC2012'
    network = 'voc-fcn8s/deploy.prototxt'
    weights = 'voc-fcn8s/fcn8s-heavy-pascal.caffemodel'
    metric = ConfusionMatrix(num_classes=21)
    net = caffe.Net(network, weights, caffe.TEST)
    for i, serial, image, label in reader(root):
        print(i, serial)
        net.blobs['data'].reshape(1, *image.shape)
        net.blobs['data'].data[...] = image
        net.forward()
        prediction = net.blobs['score'].data[0].argmax(axis=0)
        metric.update_hist(prediction, label)
    metric.report()


if __name__ == '__main__':
    main()
