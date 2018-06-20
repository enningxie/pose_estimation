import argparse
import logging
import sys
import time

from tf_pose import common
import cv2
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import matplotlib.pyplot as plt
import os

logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    parser.add_argument('--image', type=str, default='/var/Data/xz/face/images_path/ch02_20180308161036/')
    parser.add_argument('--saved_image', type=str, default='/var/Data/xz/face/pose/ch02_20180308161036/')
    parser.add_argument('--model', type=str, default='cmu', help='cmu / mobilenet_thin')

    parser.add_argument('--resize', type=str, default='1312x736',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    args = parser.parse_args()

    w, h = model_wh(args.resize)
    if w == 0 or h == 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    for image_name in os.listdir(args.image):
        print('processing {}.'.format(image_name))
        image_path = os.path.join(args.image, image_name)
        image = common.read_imgfile(image_path, None, None)
        if image is None:
            logger.error('Image can not be read, path=%s' % image_path)
            sys.exit(-1)
        t = time.time()
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        elapsed = time.time() - t
        logger.info('inference image: %s in %.4f seconds.' % (args.image, elapsed))

        image, centers = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        saved_path = os.path.join(args.saved_image, image_name)
        plt.imsave(saved_path, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))