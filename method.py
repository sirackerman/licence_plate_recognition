import cv2
import pytesseract
import numpy as np
import glob
import matplotlib.pyplot as plt

def license_plate(img_path ='D:\Desktop\ECE\Machine vision and Intelligent systems\Projects\project 1\vehicle plate\c2.jpg'):
    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_blur = cv2.GaussianBlur(hsv, [5, 5], 0)

    img_mask = cv2.inRange(hsv_blur, np.array([100, 115, 115]), np.array([124, 255, 255]))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img_lcs = cv2.morphologyEx(img_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
    img_lcs = cv2.morphologyEx(img_lcs, cv2.MORPH_CLOSE, kernel, iterations=2)

    return img_lcs

def preprocess(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, [5, 5], 5)

    _, img_thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 9))
    img_open = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 10))
    img_dilated = cv2.dilate(img_open, kernel, iterations=1)

    return img_open, img_dilated

def split_character(lcs_char, lcs_char_shape):
    contours, hierarchy = cv2.findContours(lcs_char_shape, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    chars = []
    for i in contours:
        rect = cv2.boundingRect(i)
        chars.append(rect)

    chars = sorted(chars, key=lambda x: x[0], reverse=False)
    char_imgs = []
    for char in chars:
        if char[3] > char[2] * 1.5 and char[3] < char[2] * 2.2:
            splited_char = lcs_char[char[1]:char[1] + char[3], char[0] + 8:char[0] + char[2] - 8]
            char_imgs.append(splited_char)

    for i, char_img in enumerate(char_imgs):
        plt.subplot(1, len(char_imgs), i + 1)
        plt.imshow(char_img, cmap='gray')
        plt.axis('off')
        plt.title(f'Char {i + 1}')

    plt.show()

    return char_imgs

def ocr_character(img):
    # Use pytesseract to perform OCR on the input image
    result = pytesseract.image_to_string(img, config='--psm 10 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

    # Return the recognized characters
    return result.strip()

if __name__ == '__main__':
    img_path = 'D:\Desktop\ECE\Machine vision and Intelligent systems\Projects\project 1\CCPD2020\CCPD2020\ccpd_green\train\0371875-90_265-194&441_530&552-530&552_201&546_194&445_520&441-0_0_3_25_25_26_25_31-148-51.jpg'
    img_bgr = cv2.imread(img_path)
    lp = license_plate(img_bgr)
    char, char_shape = preprocess(lp)
    char_imgs = split_character(char, char_shape)

    recognized_characters = ""
    for i, char_img in enumerate(char_imgs):
        recognized_characters += ocr_character(char_img)

    print("Recognized Plate:", recognized_characters)
