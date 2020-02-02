import numpy as np
import cv2
#import pandas as pd
import pytesseract
#from table import Table
#import ocrutils
from difflib import SequenceMatcher
#import os
import argparse

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def search_cell(form_data, key_str):
    sim_dist = -1
    key_cell = ""
    for item in form_data:
        temp_cost = similar(item[0].lower(), key_str)
        if temp_cost > sim_dist:
            key_cell = item
            sim_dist = temp_cost
    return key_cell


#to be improved
def parse_form_data(form_data, img, max_box):

    social_security_number = search_cell(form_data, "social security")[0].split("\n")[-1]

    wages_tip = search_cell(form_data, "other compensation")[0].split("\n")[-1]

    employee_address = search_cell(form_data, "zip code")[0].split("\n")[1:]

    employee_first_name = search_cell(form_data, "first name")[0].split("\n")[-1]

    year = ""
    x, y, w, h = max_box
    bottom_img = img[y+h:img.shape[0], int(img.shape[1]*0.3):int(img.shape[1]*0.7)]

    year_str = pytesseract.image_to_string(bottom_img)
    words = year_str.split(" ")
    for w in words:
        if w.isdigit():
            year = w

    return {"social_security_number": social_security_number, "wages_tip": wages_tip, "employee_address": employee_address, "employee_first_name": employee_first_name, "year": year}

def isolate_lines(src, structuring_element):
	cv2.erode(src, structuring_element, src, (-1, -1)) # makes white spots smaller
	cv2.dilate(src, structuring_element, src, (-1, -1)) # makes white spots bigger


def main(file_path):
    print("File processing ...")
    img = cv2.imread(file_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray)
    ret2, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    filtered = th2

    SCALE = 15

    # Isolate horizontal and vertical lines using morphological operations
    horizontal = filtered.copy()
    vertical = filtered.copy()
    #ocrutils.display(horizontal, "Horizontal Image")
    #ocrutils.display(vertical, "Vertical Image")

    horizontal_size = int(horizontal.shape[1] / SCALE)
    horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    isolate_lines(horizontal, horizontal_structure)

    vertical_size = int(vertical.shape[0] / SCALE)
    vertical_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    isolate_lines(vertical, vertical_structure)

    struct = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    horizontal = cv2.dilate(horizontal, struct, anchor=(-1, -1), iterations=1)
    vertical = cv2.dilate(vertical, struct, anchor=(-1, -1), iterations=1)

    grid_mask = horizontal + vertical

    # ocrutils.display(horizontal, "Horizontal")
    # ocrutils.display(vertical, "Vertical")
    #ocrutils.display(grid_mask, "Horizontal and Vertical")

    contours, hierarchy = cv2.findContours(grid_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    height, width = img.shape[0:2]
    cnts = []
    min_text_height_limit = 50
    max_text_height_limit = 35

    form_cell = []
    max_cnt_size = -1
    max_box = 0

    #maskout the possible tabular region with grid shapes
    for idx, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        if w+h > max_cnt_size:
            max_cnt_size = w+h
            max_box = (x, y, w, h)

        if idx >= len(contours)-1:
            break
        img_clone = img.copy()
        cv2.drawContours(img_clone, [cnt], -1, (0, 255, 0), 6)
        #ocrutils.display(img_clone)
        cropped_cell = img[y:y+h, x:x+w]
        ocr_str = pytesseract.image_to_string(cropped_cell)
        print("\n\nForm cell text: {}".format(idx))
        print(ocr_str)
        #ocrutils.display(cropped_cell)
        form_cell.append([ocr_str, (x, y, w, h)])


    form_data = parse_form_data(form_cell, img, max_box)

    print("\n\nForm Data")

    for key, data in form_data.items():
        print("{} : {}".format(key, data))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Use this script to extract form data from document Images')
    # Input argument
    parser.add_argument('--input', help='Path to input image file')
    #parser.add_argument('--output', help='Path to output image file')
    args = parser.parse_args()
    inputfilepath = args.input
    if (inputfilepath == None):
        print("Please provide the input file path")
        exit()

    #inputfilepath = "data/img_1.png"
    main(inputfilepath)
