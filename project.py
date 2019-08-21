import os
import sys
import cv2
import gdal
import numpy as np
import shutil
from PIL import Image
import tensorflow as tf
from tensorflow.contrib import slim
from inception_resnet_v2 import inception_resnet_v2_arg_scope, inception_resnet_v2
import time


start_time = time.time()

def judge(gray, row, col, size):
    row_lower = row - size
    row_higher = row + size
    col_lower = col - size
    col_higher = col + size
    if row_lower < 0:
        row_lower = 0
    if col_lower < 0:
        col_lower = 0
    if row_higher > 2999:
        row_higher = 2999
    if col_higher > 2999:
        col_higher = 2999
    if np.sum(gray[row_lower:row_higher, col_lower:col_higher] < 170) > 960:
        return True
    else:
        return False

data = "data"
tif = "tif"
patches = "patches"
if os.path.exists(os.path.join(data, patches)):
    shutil.rmtree(os.path.join(data, patches))
size = 3000
win = 1000
gdal.AllRegister()
num = 0
num1 = 0
num2 = 0
for image in os.listdir(os.path.join(data, tif)):
    tif = gdal.Open(os.path.join(os.path.join(data, tif), image))
    width = size
    height = size
    width_ = tif.RasterXSize
    height_ = tif.RasterYSize
    sum = (int((width_-width)/win)+1) * (int((height_-height)/win)+1)
    print("This file will be cut into %d patches" % sum)
    while width < width_:
        while height < height_:
            newData = np.zeros([size, size, 3])
            band = tif.GetRasterBand(1)
            r = band.ReadAsArray(width - size, height - size, size, size)
            band = tif.GetRasterBand(2)
            g = band.ReadAsArray(width - size, height - size, size, size)
            band = tif.GetRasterBand(3)
            b = band.ReadAsArray(width - size, height - size, size, size)
            newData[:, :, 0] = r
            newData[:, :, 1] = g
            newData[:, :, 2] = b
            image = cv2.merge([b, g, r])
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            i = 0
            for row in range(0, size, 20):
                for col in range(0, size, 20):
                    if judge(gray, row, col, 40):
                        i = i+1
            if i < 800:
                num2 += 1
                height += win
                sys.stdout.write('\r>> Converting image %d/%d' % (num + num1 + num2, sum))
                sys.stdout.flush()
                continue
            if not os.path.exists(os.path.join(data, patches)):
                os.makedirs(os.path.join(data, patches))
            if i > 14000:
                cv2.imwrite(os.path.join(os.path.join(data, patches), str(num) + "_" + str(i) + ".jpg"), image)
                num += 1
            else:
                num1 += 1
            sys.stdout.write('\r>> Converting image %d/%d' % (num + num1 + num2, sum))
            sys.stdout.flush()
            height += win
        height = size
        width += win
print("\nFinished converting all the files!")


image_pixels = 299
test = "test"
if os.path.exists(os.path.join(data, test)):
    shutil.rmtree(os.path.join(data, test))
if not os.path.exists(os.path.join(data, test)):
    os.makedirs(os.path.join(data, test))
for file in os.listdir(os.path.join(data, patches)):
    image = cv2.imread(os.path.join(os.path.join(data, patches), file), -1)
    image = cv2.resize(image, (image_pixels, image_pixels))
    cv2.imwrite(os.path.join(data, os.path.join(test, file)), image)


classes = 2
malignant = "malignant"
normal = "normal"

images = tf.placeholder(tf.float32, [None, image_pixels, image_pixels, 3], name="input/x_input")

with slim.arg_scope(inception_resnet_v2_arg_scope()):
    logits, end_points = inception_resnet_v2(images, num_classes=classes, is_training=False)


with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state("ckpt")
    if ckpt:
        print(ckpt.model_checkpoint_path)
        tf.train.Saver().restore(sess, ckpt.model_checkpoint_path)
    else:
        raise ValueError('The ckpt file is None.')
    if os.path.exists(os.path.join(data, malignant)):
        shutil.rmtree(os.path.join(data, malignant))
    if not os.path.exists(os.path.join(data, malignant)):
        os.makedirs(os.path.join(data, malignant))
    if os.path.exists(os.path.join(data, normal)):
        shutil.rmtree(os.path.join(data, normal))
    if not os.path.exists(os.path.join(data, normal)):
        os.makedirs(os.path.join(data, normal))
    for file in os.listdir(os.path.join(data, test)):
        image = os.path.join(os.path.join(data, test), file)
        img = Image.open(image)
        img = tf.decode_raw(img.tobytes(), tf.uint8)
        img = tf.reshape(img, [image_pixels, image_pixels, 3])
        img = tf.expand_dims(img, 0)
        img = tf.cast(img, tf.float32)
        result = sess.run(tf.argmax(end_points['Predictions'], 1), feed_dict={images: img.eval()})
        print(file, result)
        if result == [0]:
            shutil.move(os.path.join(os.path.join(data, patches), file), os.path.join(data, malignant), file)
        else:
            shutil.move(os.path.join(os.path.join(data, patches), file), os.path.join(data, normal), file)

end_time = time.time()
duration = end_time - start_time
print("duration:", duration)