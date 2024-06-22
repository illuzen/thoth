import cv2 
import pytesseract
import numpy as np
import re
import io
import fitz  # PyMuPDF
from glob import glob
import spacy
import contextualSpellCheck
import logging
import os

# Configure the logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger(__name__)


nlp = spacy.load('en_core_web_sm')
contextualSpellCheck.add_to_pipe(nlp)

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)


#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


#dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)


#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)


#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)


#skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


#template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)


def detect_orientation(img):
    osd = pytesseract.image_to_osd(img)
    angle = re.search('(?<=Rotate: )\d+', osd).group(0)
    # script = re.search('(?<=Script: )\d+', osd).group(0)
    if angle != '0':
        logger.info("angle: " + angle)
    # if angle != 0:
    #     cv2.rotate()
    # print("script: ", script)


def clean_string(text):
    ascii = text.encode('ascii', 'ignore').decode('ascii')
    return re.sub(r'[|><:\[\]`\-\\©=#{};!«°$€0-9*\n\'()~\"/@%&^+_]', '', ascii)


def pdf2str():
    pdfs = glob('./docs/pdf/*.pdf')
    for i, pdf in enumerate(pdfs):
        file_name = pdf.split('/')[-1].split('.')[0]
        destination = './docs/txt/{}.txt'.format(file_name)
        if os.path.exists(destination):
            logger.info('Skipping file {} as we already have a txt file for it'.format(destination))
            continue

        logger.info('Handling {}: {} / {}'.format(pdf, i, len(pdfs)))
        doc = fitz.open(pdf)
        strs = []
        # Iterate through each page
        for page_num in range(len(doc)):
            logger.info('Processing page {} of {}'.format(page_num, pdf))
            page = doc.load_page(page_num)  # number of page
            pix = page.get_pixmap()  # render page to an image
            # img = Image.open(io.BytesIO(pix.tobytes()))  # convert the image to PIL format
            stream = io.BytesIO(pix.tobytes())
            img_bytes = np.asarray(bytearray(stream.read()), dtype=np.uint8)
            img = cv2.imdecode(img_bytes, -1)  # convert the image to PIL format
            logger.info('Performing OCR on page {} of {} for {}'.format(page_num, len(doc), pdf))
            text = img2str(img)  # Perform OCR
            strs.append(text)

        assembled = ' '.join(strs)
        cleaned = clean_string(assembled)
        logger.info('Performing spell correction on {}'.format(pdf))

        try:
            if len(cleaned) > 1000000:
                half = len(cleaned) / 2
                corrected = nlp(cleaned[0:half])._.outcome_spellCheck + nlp(cleaned[half:]._.outcome_spellCheck)
            else:
                corrected = nlp(cleaned)._.outcome_spellCheck
        except Exception as e:
            logger.error('Unable to spell correct {}: {}'.format(pdf, e))
            continue

        logger.info('Spell corrected: {}'.format(corrected))

        with open(destination, 'w') as f:
            f.write(corrected)

        doc.close()


def img2str(img):

    # img = cv2.imread('image.png')
    # detect_orientation(img)
    # desk = deskew(img)
    gray = get_grayscale(img)
    # thresh = thresholding(img)
    # open = opening(gray)
    # ero = erode(gray)

    # Adding custom options
    custom_config = r'--oem 3 --psm 6'
    s = pytesseract.image_to_string(gray, config=custom_config)
    return s

pdf2str()