# encoding: UTF-8

from tombstone.core.metric import ConfusionMatrix
import tombstone.coffin as coffin
from PIL import Image
import numpy as np
import os


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
    return label


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


def parse_arg():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, nargs='?')
    ap.add_argument('--vocdevkit', type=str, nargs='?')
    ap.add_argument('--network', type=str, nargs='?')
    ap.add_argument('--weights', type=str, nargs='?')
    ap.add_argument('--num_classes', type=int, nargs='?')
    ap.add_argument('--scores', type=str, nargs='+', default=['score'])
    return ap.parse_args()


def main():
    args = parse_arg()
    root = args.vocdevkit or 'data/pascal/VOCdevkit/VOC2012'
    network = args.network or 'voc-fcn8s/deploy.prototxt'
    weights = args.weights or 'voc-fcn8s/fcn8s-heavy-pascal.caffemodel'
    num_classes = args.num_classes or 21
    if args.gpu is not None:
        coffin.use_gpu(args.gpu)
    else:
        coffin.use_cpu()
    metrics = {score: ConfusionMatrix(num_classes) for score in args.scores}
    model = coffin.Model(prototxt=network, weights=weights, phase=coffin.Phase.TEST)
    for i, serial, image, label in reader(root):
        print(i, serial)
        model.feed(data=image)
        for score_blob, metric in metrics.items():
            prediction = model[score_blob][0].argmax(axis=0)
            metric.update_hist(prediction, label)
    for score_blob, metric in metrics.items():
        print(score_blob)
        metric.report(indent=4)


if __name__ == '__main__':
    main()
