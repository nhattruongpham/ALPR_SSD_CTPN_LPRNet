import cv2
import os


def add_padding(image):
    width_size = 400
    ratio = (width_size / float(image.shape[0]))
    height_size = int((float(image.shape[1]) * float(ratio)))
    top = width_size//2
    bottom = width_size//2
    left = height_size//2
    right = height_size//2
    color = [255, 255, 255]
    img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img

def resize_im(im, scale, max_scale=None):
    f = float(scale) / min(im.shape[0], im.shape[1])
    if max_scale != None and f * max(im.shape[0], im.shape[1]) > max_scale:
        f = float(max_scale) / max(im.shape[0], im.shape[1])
    return cv2.resize(im, None, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR), f


def get_file_list(path):

    # TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 20) ]
    TEST_IMAGE_PATHS = []
    for root, directories, filenames in os.walk(path):
        for file in filenames:
            if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
                TEST_IMAGE_PATHS.append(os.path.join(root, file))
    # print(TEST_IMAGE_PATHS)
    return TEST_IMAGE_PATHS