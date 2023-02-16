import pytesseract
import cv2
from gtts import gTTS
from playsound import playsound
import numpy as np
import matplotlib.pyplot as plt
import sys

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

def drawROI(img, corners):
    cpy = img.copy()

    c1 = (192, 192, 255)
    c2 = (128, 128, 255)

    for pt in corners:
        cv2.circle(cpy, tuple(pt.astype(int)), 25, c1, -1, cv2.LINE_AA)

    cv2.line(cpy, tuple(corners[0].astype(int)), tuple(corners[1].astype(int)), c2, 2, cv2.LINE_AA)
    cv2.line(cpy, tuple(corners[1].astype(int)), tuple(corners[2].astype(int)), c2, 2, cv2.LINE_AA)
    cv2.line(cpy, tuple(corners[2].astype(int)), tuple(corners[3].astype(int)), c2, 2, cv2.LINE_AA)
    cv2.line(cpy, tuple(corners[3].astype(int)), tuple(corners[0].astype(int)), c2, 2, cv2.LINE_AA)

    disp = cv2.addWeighted(img, 0.3, cpy, 0.7, 0)

    return disp

def avg_blur(img1, kernel_size=(5,5)):
    return cv2.blur(img1, kernel_size)

def onMouse(event, x, y, flags, param):
    global srcQuad, dragSrc, ptOld, src

    if event == cv2.EVENT_LBUTTONDOWN:
        for i in range(4):
            if cv2.norm(srcQuad[i] - (x, y)) < 25:
                dragSrc[i] = True
                ptOld = (x, y)
                break

    if event == cv2.EVENT_LBUTTONUP:
        for i in range(4):
            dragSrc[i] = False

    if event == cv2.EVENT_MOUSEMOVE:
        for i in range(4):
            if dragSrc[i]:
                dx = x - ptOld[0]
                dy = y - ptOld[1]

                srcQuad[i] += (dx, dy)

                cpy = drawROI(img1, srcQuad)
                cv2.imshow('img', cpy)
                ptOld = (x, y)
                break

img1 = cv2.imread('Capture_1.jpg')

if img1 is None:
    print('Image open failed!')
    sys.exit()

h, w = img1.shape[:2]
dw = 500
dh = round(dw * 297 / 210)

srcQuad = np.array([[30, 30], [30, h-30], [w-30, h-30], [w-30, 30]], np.float32)
dstQuad = np.array([[0, 0], [0, dh-1], [dw-1, dh-1], [dw-1, 0]], np.float32)
dragSrc = [False, False, False, False]

disp = drawROI(img1, srcQuad)

cv2.imshow('img', disp)
cv2.setMouseCallback('img', onMouse)

while True:
    key = cv2.waitKey()
    if key == 13:
        break
    elif key == 27:
        cv2.destroyWindow('img')
        sys.exit()

pers = cv2.getPerspectiveTransform(srcQuad, dstQuad)
dst = cv2.warpPerspective(img1, pers, (dw, dh), flags=cv2.INTER_CUBIC)

img_f = img1.astype(np.float32)
img_norm = ((img_f - img_f.min()) * (255) / (img_f.max() - img_f.min()))
img_norm = img_norm.astype(np.uint8)

img_norm2 = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX)

hist = cv2.calcHist([img1], [0], None, [256], [0, 255])
hist_norm = cv2.calcHist([img_norm], [0], None, [256], [0, 255])
hist_norm2 = cv2.calcHist([img_norm2], [0], None, [256], [0, 255])

hists = {'Before' : hist, 'Manual':hist_norm, 'cv2.normalize()':hist_norm2}
for i, (k, v) in enumerate(hists.items()):
    plt.subplot(1,3,i+1)
    plt.title(k)
    plt.plot(v)
plt.show()

h1Img, w1Img, _ = img1.shape

box1 = pytesseract.image_to_boxes(img1, lang='kor')

data1 = pytesseract.image_to_data(img1, lang='kor')

filewrite = open("string.txt", "w")

for z, a in enumerate(data1.splitlines()):
    if z != 0:
        a = a.split()
        if len(a) == 12:
            x, y = int(a[6]), int(a[7])
            w, h = int(a[8]), int(a[9])
            cv2.rectangle(img1, (x, y), (x + w, y + h), (255, 0, 0), 1)
            filewrite.write(a[11] + " ")

filewrite.close()
fileread = open("string.txt", "r")
lang = 'ko'
line = fileread.read()

if line != ' ':
    speech = gTTS(text= line, lang= lang, slow= False)
    speech.save("test.mp3")
cv2.imshow('gtts', img1)
cv2.waitKey(0)
playsound("test.mp3")
