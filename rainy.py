import time
import cv2
import numpy as np
import os
import random


def get_noise(img_shape):
    noise = np.random.uniform(0, 256, img_shape)
    noise[np.where(noise < 250)] = 0
    c = np.array([[0, 0.1, 0], [0.1, 8, 0.1], [0, 0.1, 0]])
    noise = cv2.filter2D(noise, -1, c)
    return noise


def rain_blur(noise):
    length = np.random.randint(60, 71)
    angle = np.random.uniform(-50, 51)

    trans = cv2.getRotationMatrix2D((length / 2, length / 2), angle - 45, 1 - length / 100.0)
    dig = np.diag(np.ones(length))
    k = cv2.warpAffine(dig, trans, (length, length))
    k = cv2.GaussianBlur(k, (5, 5), 0)

    blurred = cv2.filter2D(noise, -1, k)
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred


def add_rain_mix(rain, img, img_path):
    beta = np.random.uniform(0.7, 0.9)
    rain = np.expand_dims(rain, 2)
    rain_result = img.copy()
    rain = np.array(rain, dtype=np.float32)
    rain_result[:, :, 0] = rain_result[:, :, 0] * (255 - rain[:, :, 0]) / 255.0 + beta * rain[:, :, 0]
    rain_result[:, :, 1] = rain_result[:, :, 1] * (255 - rain[:, :, 0]) / 255 + beta * rain[:, :, 0]
    rain_result[:, :, 2] = rain_result[:, :, 2] * (255 - rain[:, :, 0]) / 255 + beta * rain[:, :, 0]

    cv2.imwrite(img_path, rain_result)


def add_rain_cover(rain, img, img_path, alpha=0.9):
    rain = np.expand_dims(rain, 2)
    rain = np.repeat(rain, 3, 2)
    result = cv2.addWeighted(img, alpha, rain, 1 - alpha, 1)

    cv2.imwrite(img_path, result)


def process_image(img_path):
    if random.random() < 0.5:  # decide whether to add rain on the image by 0.5 prob
        img = cv2.imread(img_path)
        noise = get_noise(img.shape[0: 2])
        blurred_noise = rain_blur(noise)
        if random.random() < 0.5:  # decide using which rain-adding method by 0.5 prob
            add_rain_mix(blurred_noise, img, img_path)
        else:
            add_rain_cover(blurred_noise, img, img_path)


def walk_and_process(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            process_image(os.path.join(root, file))


start_time = time.time()
walk_and_process('data/cityscapes/train')
walk_and_process('data/cityscapes/val')
end_time = time.time()
print(f"Rain process duration: {end_time - start_time}")