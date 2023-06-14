# import required libraries
import cv2
import os
import numpy as np


def frontface_detection(image):
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(image, 1.2, 2)

    return faces


def profileface_detection(image):
    face_cascade = cv2.CascadeClassifier("haarcascade_profileface.xml")
    faces = face_cascade.detectMultiScale(image, 1.5, 2)

    return faces

def add_noise(image, epsilon, m, b):
    sensitivity = (255 * m) / (b ** 2)
    noise = np.random.laplace(0, sensitivity / epsilon, image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    return noisy_image

def blurring(image, file_name, faces, k, eps, b0):
    blur_img = image.copy()

    for x, y, w, h in faces:
        roi = image[y : y + h, x : x + w]

        # apply DP-pix with size b0
        for i in range(0, roi.shape[0], b0):
            for j in range(0, roi.shape[1], b0):
                grid = roi[i: i + b0, j: j + b0]
                avg_pixel = np.mean(grid, axis=(0, 1))
                avg_pixel = add_noise(avg_pixel, eps, 50, b0)
                roi[i: i + b0, j: j + b0] = avg_pixel.astype(np.uint8)

        # upsample to original size
        temp = cv2.resize(roi, (roi.shape[0] * 2, roi.shape[1] * 2), interpolation=cv2.INTER_NEAREST)
        roi = cv2.resize(temp, (roi.shape[0], roi.shape[1]), interpolation=cv2.INTER_NEAREST)

        # apply gaussian blur to face rectangle
        roi = cv2.GaussianBlur(roi, (k, k), k)

        # add blurred face on original image to get final image
        blur_img[y : y + roi.shape[0], x : x + roi.shape[1]] = roi

    # save image
    cv2.imwrite(f"dp/blur/{k}/{eps}/{file_name}", blur_img)


def pixelization(image, file_name, faces, size, eps):
    pixel_img = image.copy()

    for x, y, w, h in faces:
        roi = pixel_img[y : y + h, x : x + w]

        # get mean of the grid
        for i in range(0, roi.shape[0], size):
            for j in range(0, roi.shape[1], size):
                grid = roi[i: i + size, j: j + size]
                avg_pixel = np.mean(grid, axis=(0, 1))
                avg_pixel = add_noise(avg_pixel, eps, 50, size)
                roi[i: i + size, j: j + size] = avg_pixel.astype(np.uint8)

        pixel_img[y : y + roi.shape[0], x : x + roi.shape[1]] = roi

    # save image
    cv2.imwrite(f"dp/pixel/{size}/{eps}/{file_name}", pixel_img)

def img_resize(img, size, padColor):
    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA
    else: # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w/h

    # compute scaling and pad sizing
    if aspect > 1: # horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1: # vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    return scaled_img

if __name__ == "__main__":
    # folder_path = "img/"
    folder_path = "att_img_flat/"

    # iterate every image in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".png"):
            # read image
            img_path = os.path.join(folder_path, file_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            #resize
            img = img_resize(img, (128,128), 0)

            # face detection
            faces = np.array([0,0,128,128]).reshape(-1,4)

            # process image
            if len(faces) != 0:
                blurring(img, file_name, faces, 15, 0.001, 4)
                blurring(img, file_name, faces, 15, 0.1, 4)
                blurring(img, file_name, faces, 15, 1, 4)

                blurring(img, file_name, faces, 45, 0.001, 4)
                blurring(img, file_name, faces, 45, 0.1, 4)
                blurring(img, file_name, faces, 45, 1, 4)

                blurring(img, file_name, faces, 99, 0.001, 4)
                blurring(img, file_name, faces, 99, 0.1, 4)
                blurring(img, file_name, faces, 99, 1, 4)

                pixelization(img, file_name, faces, 4, 0.001)
                pixelization(img, file_name, faces, 4, 0.1)
                pixelization(img, file_name, faces, 4, 1)

                pixelization(img, file_name, faces, 8, 0.001)
                pixelization(img, file_name, faces, 8, 0.1)
                pixelization(img, file_name, faces, 8, 1)

                pixelization(img, file_name, faces, 16, 0.001)
                pixelization(img, file_name, faces, 16, 0.1)
                pixelization(img, file_name, faces, 16, 1)

                print(file_name+" finished")
