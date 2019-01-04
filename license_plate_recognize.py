import glob
import os
import shutil
import sys

import time

import cv2
import numpy as np
import tensorflow as tf
sys.path.append('.')

from lib.detect.text_2_line_detector import *
from lib.detect.plateDetector import *
from lib.tools import add_padding, resize_im, get_file_list
from lib.lprNet.run_lprnet import *

plate_number1 = {}
plate_number2 = {}

def get_low_box(coods):
    temp = []
    for i in range(0, len(coods)):
        box = coods[i]
        y_max = max(int(box[1]), int(box[3]))
        temp.append(y_max)
    if temp:
        return temp.index(max(temp))
    return temp


def save_image_full(image, name):
    cv2.imwrite(os.path.join(os.path.join(os.getcwd(), "data/result_full/" + name)), image)

def save_image_low(image, name):
    cv2.imwrite(os.path.join(os.path.join(os.getcwd(), "data/result_low/" + name)), image)

if __name__ == '__main__':
    path = './data/demo'
    im_names = get_file_list(path)
    for im_name in im_names:
        img = cv2.imread(im_name) #image
        detect_output_full = get_license_plate_full(img, detection_graph_full)

        print("*")
        for coordinate_full, label in detect_output_full.items():
            full_license_plate_image = img[coordinate_full[1]:coordinate_full[3], coordinate_full[0]:coordinate_full[2]]

            pad_full_license_plate_image = add_padding(full_license_plate_image)
            detect_output_low1 = get_license_plate_low(pad_full_license_plate_image, detection_graph_low)
            if len(detect_output_low1.values()) == 2:
                multi_box1 = []
                for coordinate_low1, label in detect_output_low1.items():
                    multi_box1.append(coordinate_low1)
                low_box1 = get_low_box(multi_box1)
                box_new1 = multi_box1[low_box1]
                license_plate_image_low1 = pad_full_license_plate_image[int(box_new1[1]):int(box_new1[3]), int(box_new1[0]):int(box_new1[2])]
                detected_list1 = extract_license_number(license_plate_image_low1)
                if detected_list1 in plate_number1:
                    current_index1 = len(plate_number1[detected_list1])
                    current_plate1 = detected_list1 + "_" + str(current_index1 + 1) + ".png"
                    plate_number1[detected_list1].append(current_plate1)
                    save_image_full(full_license_plate_image, current_plate1)
                else:
                    current_plate1 = detected_list1 + ".png"
                    plate_number1[detected_list1] = [current_plate1]
                    save_image_full(full_license_plate_image, current_plate1)

                del license_plate_image_low1

            else:
                detected_list1 = ''

        detect_output_low = get_license_plate_low(img, detection_graph_low)
        if len(detect_output_low.values()) == 2:
            multi_box = []
            for coordinate_low, label in detect_output_low.items():
                multi_box.append(coordinate_low)

            low_box = get_low_box(multi_box)
            box_new = multi_box[low_box]
            license_plate_image_low = img[int(box_new[1]):int(box_new[3]), int(box_new[0]):int(box_new[2])] #image
            detected_list2 = extract_license_number(license_plate_image_low)
            if detected_list2 in plate_number2:
                current_index2 = len(plate_number2[detected_list2])
                current_plate2 = detected_list2 + "_" + str(current_index2 + 1) + ".png"
                plate_number2[detected_list2].append(current_index2)
                save_image_low(license_plate_image_low, current_plate2)
            else:
                current_plate2 = detected_list2 + ".png"
                plate_number2[detected_list2] = [current_plate2]
                save_image_low(license_plate_image_low, current_plate2)

            del license_plate_image_low

        else:

            detected_list2 = ''

        if (len(detected_list2) > 3) and (len(detected_list2) < 6):
            detected_list = detected_list2
            print(detected_list)
        elif (len(detected_list2) < 4 or len(detected_list2) > 5):
            detected_list = detected_list1
            print(detected_list)




