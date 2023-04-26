# import required libraries
import cv2
import os
import numpy as np
import os
from PIL import Image

def frontface_detection(image):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, 1.2, 2)

    return faces

def profileface_detection(image):
    face_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')
    faces = face_cascade.detectMultiScale(image, 1.5, 2)

    return faces

def blurring(image, file_name, faces, size):
    blur_img = image.copy()

    for (x, y, w, h) in faces:
        # apply gaussian blur to face rectangle
        roi = blur_img[y:y+h, x:x+w]
        roi = cv2.GaussianBlur(roi, (size, size), size)
 
        # add blurred face on original image to get final image
        blur_img[y:y+roi.shape[0], x:x+roi.shape[1]] = roi

    # save image
    
    cv2.imwrite(f'blur/{size}/{file_name}', blur_img)

def pixelization(image, file_name, faces, size):
    pixel_img = image.copy()

    for (x, y, w, h) in faces:
        roi = pixel_img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (w // size, h // size), interpolation=cv2.INTER_LINEAR)
        roi = cv2.resize(roi, (w, h), interpolation=cv2.INTER_NEAREST)

        pixel_img[y:y+roi.shape[0], x:x+roi.shape[1]] = roi

    # 儲存圖片
    cv2.imwrite(f'pixel/{size}/{file_name}', pixel_img)

def img_resize(img, size, padColor):
    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA
    else: # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w/h  # if on Python 2, you might need to cast as a float: float(w)/h

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

if __name__ == '__main__':
    folder_path = "img/"
    folder_path = "att_img_flat/"
    count = 0
    num = 0

    # iterate every image in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".png"):
            # read image
            img_path = os.path.join(folder_path, file_name)
            img = cv2.imread(img_path)
            num += 1

            #resize
            img = img_resize(img, (300,300), 0)

            #face detection
            faces = np.append(np.asarray(frontface_detection(img)).reshape(-1, 4), np.asarray(profileface_detection(img)).reshape(-1, 4), axis=0)
            faces = faces.astype(int)

            # process image
            if len(faces) != 0:
                if(img.shape[0]==300 and img.shape[1]==300):
                    blurring(img, file_name, faces, 15)
                    blurring(img, file_name, faces, 45)
                    blurring(img, file_name, faces, 99)
                    pixelization(img, file_name, faces, 4)
                    pixelization(img, file_name, faces, 8)
                    pixelization(img, file_name, faces, 16)
                    count += 1
                else:
                    print("err\n")


    print(f'success detection percentage: {count/num}')