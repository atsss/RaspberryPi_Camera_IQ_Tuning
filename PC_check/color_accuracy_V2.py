import cv2
from PIL import Image
import numpy as np
import time
import os, sys
from colormath.color_diff import delta_e_cie2000 , delta_e_cie1976
from colormath.color_objects import sRGBColor ,LabColor
from colormath.color_conversions import convert_color

print(np.__version__)
print(cv2.__version__)
#print(colour.__version__)

IMG_SIZE = ((int)(3280/1.5), (int)(2464/1.5))
GOPRO_FILE_NAME = 'GOPR0075.jpg'
IMX_FILE_NAME = 'image_json.jpg'



def plot_rgbl(img_rgbl, box_size = (25, 25)):
    nb_boxes = img_rgbl.shape[0]

    if len(img_rgbl.shape) == 3:
        clrs = np.ascontiguousarray(np.tile(img_rgbl[:, 0, 0:3].reshape(nb_boxes, 3, 1, 1), (box_size[0], box_size[1])).transpose(0, 2, 3, 1))
    elif len(img_rgbl.shape) == 2:
        clrs = np.ascontiguousarray(np.tile(img_rgbl[:, 0:3].reshape(nb_boxes, 3, 1, 1), (box_size[0], box_size[1])).transpose(0, 2, 3, 1))
    else:
        raise ValueError

    clrs.resize((nb_boxes*box_size[0], box_size[1], 3))
    clrs = clrs.transpose(1, 0, 2)
    clrs = np.clip(clrs * 255, 0, 255).astype('uint8')
    return clrs
def plot_rgblx2(img_rgbl_1, img_rgbl_2 ,box_size = (25, 25)):
    clrs = np.zeros((box_size[0]*2, box_size[1]*24,3))
    for j in range(box_size[0]):
        for i in range(box_size[1]*24):
            clrs[j][i] =  img_rgbl_1[(int)(i/box_size[1])]

    for j in range(box_size[0],box_size[0]*2):
        for i in range(box_size[1]*24):
            clrs[j][i] =  img_rgbl_2[(int)(i/box_size[1])]

    print(clrs.shape)
    clrs = np.clip(clrs * 255, 0, 255).astype('uint8')
    print(clrs.shape)
    return clrs

def crop_center(image, crop_width, crop_height):
    width, height = image.size

    # トリミングする領域の左上の座標を計算
    # left = int((width - crop_width) / 2)
    left = int((width - crop_width) / 2 - 50)
    # top = int((height - crop_height) / 2)
    top = int((height - crop_height) / 2 - 50)

    # 画像をトリミング
    cropped_image = image.crop((left, top, left + crop_width, top + crop_height))

    return cropped_image

def rgb_to_lab(rgb_normalized):
    # RGB の値を 0-1 の範囲に正規化
    # rgb_normalized = [channel / 255.0 for channel in rgb]

    # RGB を XYZ に変換
    xyz = colour.sRGB_to_XYZ(rgb_normalized)

    # XYZ を Lab に変換
    lab = colour.XYZ_to_Lab(xyz)

    return lab
def colormath_rgb2lab(rgb):
    return convert_color(sRGBColor(rgb[0][0],rgb[0][1],rgb[0][2]), LabColor, target_illuminant='d65')


FILE_PATH = ".\img" #os.path.join('/', 'Users', 'ats', 'hobbies', 'notebooks', 'assets')
ILUMINATIONS = ['daylightLow', ]
DEVICES = ['gopro', 'imx',]
IMGS = {}

for ilumination in ILUMINATIONS:
    IMGS[ilumination] = {}
    for device in DEVICES:
        if device == 'gopro':
            filepath = GOPRO_FILE_NAME
            print(filepath)
            assert os.path.isfile(GOPRO_FILE_NAME)
            img = Image.open(filepath).rotate(0) # Rotate to position the WHITE on the bottom left corner
            IMGS[ilumination][device] = crop_center(img, IMG_SIZE[0], IMG_SIZE[1])
        else:
            filepath = IMX_FILE_NAME
            print(filepath)
            assert os.path.isfile(filepath)
            # img = Image.open(filepath).rotate(180) # Rotate to position the WHITE on the bottom left corner
            img = Image.open(filepath).rotate(180) # Rotate to position the WHITE on the bottom left corner
            IMGS[ilumination][device] = crop_center(img, IMG_SIZE[0], IMG_SIZE[1])


for ilumination in ILUMINATIONS:
    print(ilumination)
    comp_img = np.zeros((IMG_SIZE[1], IMG_SIZE[0]*len(DEVICES),3), 'uint8')
    for i, device in enumerate(DEVICES):
        img = np.array(IMGS[ilumination][device])
        comp_img[:, IMG_SIZE[0]*i:IMG_SIZE[0]*(i+1), :] = img
    Image.fromarray(comp_img).show()

CHARTS = {}
CHECKER_SRCS = {}
for ilumination in ILUMINATIONS:
    CHARTS[ilumination] = {}
    CHECKER_SRCS[ilumination] = {}
    print(ilumination)
    comp_img = np.zeros((IMG_SIZE[1], IMG_SIZE[0]*len(DEVICES), 3), 'uint8')
    srcs = []
    for i, device in enumerate(DEVICES):
        img = np.array(IMGS[ilumination][device]).copy()

        # Detect Color CHart and extract checker
        detector = cv2.mcc.CCheckerDetector_create()
        succ = detector.process(img, chartType=cv2.mcc.MCC24, nc=1, useNet=True)
        assert succ
        checker = detector.getBestColorChecker()
        CHARTS[ilumination][device] = checker

        # Process checker to extract colors
        chartsRGB = checker.getChartsRGB()
        src_rgb = chartsRGB[:, 1].copy()

        # Bottom left corner is assumed to be WHITE
        src_rgb = np.round(src_rgb, 0).astype('uint8')
        src_rgb.resize(24, 3)
        src_rgb = src_rgb[::-1, ::-1]
        src = (src_rgb.copy()/255)[:, np.newaxis, :]

        # Check direction
        if src[0].mean() < src[-1].mean():
            src = src[::-1]
        CHECKER_SRCS[ilumination][device] = src

        # Keep to plat later
        srcs.append(src)

        # Draw
        draw = cv2.mcc.CCheckerDraw_create(checker, color=(0, 255, 0), thickness=2)
        draw.draw(img)

        comp_img[:, IMG_SIZE[0]*i:IMG_SIZE[0]*(i+1), :] = img
    #display(Image.fromarray(comp_img))
    Image.fromarray(comp_img).show()

    # Plot colors
    #for src in srcs:
        #display(Image.fromarray(plot_rgbl(src.copy(), (50, 50))))
        #Image.fromarray(plot_rgbl(src.copy(), (50, 50))).show()
    Image.fromarray(plot_rgblx2(srcs[0],srcs[1], (50, 50))).show()


GOPRO_INDEX = 0
IMX_INDEX = 1

normalized_rgbs = np.array(srcs)
rows = len(normalized_rgbs)
columns = len(normalized_rgbs[GOPRO_INDEX])

assert rows == 2
assert columns == 24

# https://www.imatest.com/docs/colorcheck
labels = [
    'dark skin', 'light skin', 'blue sky', 'foliage', 'blue flower', 'bluish green',
    'orange', 'purplish blue', 'moderate red', 'purple', 'yellow green', 'orange yellow',
    'blue', 'green', 'red', 'yellow', 'magenta', 'cyan',
    'white', 'neutral 8', 'neutral 6.5', 'neutral 5', 'neutral 3.5', 'black'
]
deltEs = np.empty(columns)
for j in range(columns):
    #target_lab = rgb_to_lab(normalized_rgbs[GOPRO_INDEX][j])
    #tuned_lab = rgb_to_lab(normalized_rgbs[IMX_INDEX][j])
    #deltE = np.sqrt(((target_lab - tuned_lab) ** 2).sum())
    #deltE = delta_e_cie1976(colormath_rgb2lab(normalized_rgbs[GOPRO_INDEX][j]), colormath_rgb2lab(normalized_rgbs[IMX_INDEX][j]))
    deltE = delta_e_cie2000(colormath_rgb2lab(normalized_rgbs[GOPRO_INDEX][j]), colormath_rgb2lab(normalized_rgbs[IMX_INDEX][j]))

    print(f'{labels[j]}: {deltE}')
    deltEs[j] = deltE

print(f'Total: {deltEs.sum()}')
print(f'Total: {deltEs.sum()/24}')

