import cv2
import streamlit
import numpy as np
import pandas as pd
from PIL import Image, ImageChops
import pytesseract
import pyocr
import pyocr.builders
import easyocr


def process(img):
    cv = pil_to_cv(img)
    # Line-less copy
    image_without_borders = remove_lines(cv)
    # Get results
    cropped_without_borders = crop_periods_of_service(image_without_borders)
    cropped_with_borders = crop_periods_of_service(cv)

    boxes, bitnot, row, countcol = locate_cells(cropped_with_borders)

    arr = analyze_cells(cropped_without_borders, boxes, bitnot)

    dataframe = pd.DataFrame(arr.reshape(len(row), countcol))

    dataframe.columns = ['Line #', 'SERVICE', 'ENL', 'WO', 'COM', 'PAY', 'AD', 'NONE',
                         'FROM: YR.', 'FROM: MO.', 'FROM: DAYS', 'TO: YR.', 'TO: MO.', 'TO: DAYS',
                         'POINTS', 'LOST TIME', 'SOURCE DOCUMENT']

    dataframe.index += 1
    dataframe = dataframe.drop(columns=['Line #'])

    return dataframe


def pil_to_cv(img):
    # For reversing the operation:
    im_np = np.asarray(img)
    return im_np


def cv_to_pil(img):
    # You may need to convert the color.
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img2)
    return im_pil


def remove_lines(form_image):
    # Thresholding the image
    (thresh, img_bin) = cv2.threshold(form_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # Invert the image
    img_bin = 255 - img_bin

    # Defining a kernel length
    kernel_length = np.array(img_bin).shape[1] // 80

    # Verticle kernel of (1 X kernel_length)
    # > detects all the verticle lines from the image.

    # Horizontal kernel of (kernel_length X 1)
    # > detects all the horizontal line from the image.
    verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))

    # A kernel of (3 X 3) ones.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Morphological operation to detect vertical lines from an image
    img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=3)
    verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)

    # Morphological operation to detect horizontal lines from an image
    img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)

    # Weighting parameters, this will decide the quantity of an image to be added to make a new image.
    alpha = 0.5
    beta = 1.0 - alpha
    # This function helps to add two image with specific weight
    # parameter to get a third image as summation of two image.
    img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
    img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
    (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    invert = cv2.bitwise_not(img_final_bin)
    dst = cv2.addWeighted(form_image, 1, invert, 1, 0)

    return dst


def has_text(df, txt):
    return len(df[df['text'] == txt]) > 0


def crop_periods_of_service(img):
    # Get results
    df = pytesseract.image_to_data(img, lang='eng', output_type='data.frame')

    # Get width and height for crop
    height, width = img.shape

    if has_text(df, 'IDENTIFICATION'):
        y = df[df['text'] == 'IDENTIFICATION']['top'].iloc[0]
        h = df[df['text'] == 'IDENTIFICATION']['height'].iloc[0]
        finder = img[y + h:height, 0:width]

    elif has_text(df, 'NUMBER'):
        y = df[df['text'] == 'NUMBER']['top'].iloc[0]
        h = df[df['text'] == 'NUMBER']['height'].iloc[0]
        finder = img[y + h:height, 0:width]

    df = pytesseract.image_to_data(img, lang='eng', output_type='data.frame')

    if has_text(df, 'PERIODS'):
        y = df[df['text'] == 'PERIODS']['top'].iloc[0]
        h = df[df['text'] == 'PERIODS']['height'].iloc[0]
        finder = img[y + int(h * 6.5):y + int(h * 70), 0:width]

    elif has_text(df, 'SERVICE'):
        y = df[df['text'] == 'SERVICE']['top'].iloc[0]
        h = df[df['text'] == 'SERVICE']['height'].iloc[0]
        finder = img[y + int(h * 6.5):y + int(h * 70), 0:width]

    return finder


def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)


def locate_cells(img):
    # Thresholding the image to a binary image
    thresh, img_bin = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Inverting the image
    img_bin = 255 - img_bin
    #     cv2.imwrite('cv_inverted.png',img_bin)

    # countcol(width) of kernel as 100th of total width
    kernel_len = np.array(img).shape[1] // 100

    # Defining a vertical kernel to detect all vertical lines of image
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))

    # Defining a horizontal kernel to detect all horizontal lines of image
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))

    # A kernel of 2x2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    # Use vertical kernel to detect and save the vertical lines in a jpg
    image_1 = cv2.erode(img_bin, ver_kernel, iterations=3)
    vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=3)
    #     cv2.imwrite("vertical.jpg",vertical_lines)

    # Use horizontal kernel to detect and save the horizontal lines in a jpg
    image_2 = cv2.erode(img_bin, hor_kernel, iterations=3)
    horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=3)
    #     cv2.imwrite("horizontal.jpg",horizontal_lines)

    # Combine horizontal and vertical lines in a new third image, with both having same weight.
    img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)

    # Eroding and thesholding the image
    img_vh = cv2.erode(~img_vh, kernel, iterations=2)
    thresh, img_vh = cv2.threshold(img_vh, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #     cv2.imwrite("img_vh.jpg", img_vh)
    bitxor = cv2.bitwise_xor(img, img_vh)
    bitnot = cv2.bitwise_not(bitxor)
    cv2.imwrite("bitnot.jpg", bitnot)

    # Detect contours for following box detection
    contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sort all the contours by top to bottom.
    contours, boundingBoxes = sort_contours(contours, method="top-to-bottom")

    # Creating a list of heights for all detected boxes
    heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]

    # Get mean of heights
    mean = np.mean(heights)

    # Create list box to store all boxes in
    box = []
    # Get position (x,y), width and height for every contour and show the contour on image
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (w < 1000 and h < 500):
            image = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            box.append([x, y, w, h])

    # Creating two lists to define row and column in which cell is located
    row = []
    column = []
    j = 0

    # Sorting the boxes to their respective row and column
    for i in range(len(box)):

        if (i == 0):
            column.append(box[i])
            previous = box[i]

        else:
            if (box[i][1] <= previous[1] + mean / 2):
                column.append(box[i])
                previous = box[i]

                if (i == len(box) - 1):
                    row.append(column)

            else:
                row.append(column)
                column = []
                previous = box[i]
                column.append(box[i])

    # calculating maximum number of cells
    countcol = 0
    for i in range(len(row)):
        countcol = len(row[i])
        if countcol > countcol:
            countcol = countcol

    # Retrieving the center of each column
    center = [int(row[i][j][0] + row[i][j][2] / 2) for j in range(len(row[i])) if row[0]]

    center = np.array(center)
    center.sort()
    # Regarding the distance to the columns center, the boxes are arranged in respective order

    finalboxes = []
    for i in range(len(row)):
        lis = []
        for k in range(countcol):
            lis.append([])
        for j in range(len(row[i])):
            diff = abs(center - (row[i][j][0] + row[i][j][2] / 4))
            minimum = min(diff)
            indexing = list(diff).index(minimum)
            lis[indexing].append(row[i][j])
        finalboxes.append(lis)

    return finalboxes, bitnot, row, countcol

    # <------ DETERMINE LAST ROW


def text_from_pyocr(img):
    tools = pyocr.get_available_tools()
    tool = tools[0]
    # PyOCR
    # Convert to PIL
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    text = tool.image_to_string(im_pil, lang='eng', builder=pyocr.builders.TextBuilder())
    return text


def text_from_easyocr(img):
    reader = easyocr.Reader(['en'])
    try:
        t = reader.readtext(img, paragraph="False")
        text = t[0][1]
    except Exception as e:
        text = ''
    return text


def image_is_blank(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)

    # Filter out noise and check if the image is blank
    image_file = im_pil.point(lambda p: 255 if p > 200 else 0)
    if ImageChops.invert(image_file).getbbox():
        return False
    else:
        return True


def analyze_cells(img1, finalboxes, bitnot):
    # from every single image-based cell/box the strings are extracted via pytesseract and stored in a list
    outer = []
    # Row iteration
    for i in range(len(finalboxes)):
        # Column iteration
        for j in range(len(finalboxes[i])):
            inner = ''
            if (len(finalboxes[i][j]) == 0):
                outer.append(' ')
            else:
                for k in range(len(finalboxes[i][j])):

                    y, x, w, h = finalboxes[i][j][k][0], finalboxes[i][j][k][1], finalboxes[i][j][k][2], \
                                 finalboxes[i][j][k][3]

                    if j not in [0, 2, 3, 4, 5, 6, 7, 14, 15]:

                        finalimg1 = img1[x:x + h, y:y + w]

                        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))

                        # Create a white border around the image.
                        border = cv2.copyMakeBorder(finalimg1, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[255, 255])

                        resizing = cv2.resize(border, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

                        dilation = cv2.dilate(resizing, kernel, iterations=1)
                        erosion = cv2.erode(dilation, kernel, iterations=2)

                        if not image_is_blank(erosion):

                            if j == 1:
                                out = text_from_easyocr(erosion)

                            else:
                                # Try PyOCR
                                out = text_from_pyocr(erosion)

                                if len(out) <= 1:
                                    if not image_is_blank(erosion):
                                        out = text_from_easyocr(erosion)

                            inner = inner + " " + out
                        else:
                            inner = inner + " "

                outer.append(inner)

    return np.array(outer)