import cv2
from PIL import Image
import numpy as np
import time
import math
import os, sys
from colormath.color_diff import delta_e_cie2000 , delta_e_cie1976
from colormath.color_objects import sRGBColor ,LabColor ,XYZColor
from colormath.color_conversions import convert_color
import numpy as np


print(np.__version__)
print(cv2.__version__)


IMG_SIZE = ((int)(3280/1.5), (int)(2464/1.5))
IMG_FILE_NAME = 'image.jpg'
GOPR_FILE_NAME = 'GOPR0075.jpg'


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

def plot_rgb_chart(rgbl, box_size = (25, 25)):
    nb_boxes = rgbl.shape[0]

    clrs = np.ascontiguousarray(np.tile(rgbl[:, 0:3].reshape(nb_boxes, 3, 1, 1), (box_size[0], box_size[1])).transpose(0, 2, 3, 1))

    clrs.resize((nb_boxes*box_size[0], box_size[1], 3))
    clrs = clrs.transpose(1, 0, 2)
    clrs = np.clip(clrs * 255, 0, 255).astype('uint8')
    return clrs

def creat_ccm_2(deltaRGBs):
    gain =   -0.22
    r_gain =  gain  + 0.00
    g_gain =  gain  + 0.00
    b_gain =  gain  + 0.00
    cnt_offset = 1.18

    #2504 FFa54f
    #3555 e3e9ff
    #4725 cfddff
    ctt_r = 0xef/255.0 *1.0
    ctt_g = 0xdd/255.0 *1.0
    ctt_b = 0xff/255.0 *1.0

    r_offset = 0.155
    g_offset = 0.185
    b_offset = 0.205

    print(np.amin(deltaRGBs[...,0]))
    sumR = 1+ (deltaRGBs[...,0].sum()/24)
    sumG = 1+ (deltaRGBs[...,1].sum()/24)
    sumB = 1+ (deltaRGBs[...,2].sum()/24)

    sumR2 = 1+((np.amax(deltaRGBs[...,0])+np.amin(deltaRGBs[...,0]))/2)
    sumG2 = 1+((np.amax(deltaRGBs[...,1])+np.amin(deltaRGBs[...,1]))/2)
    sumB2 = 1+((np.amax(deltaRGBs[...,2])+np.amin(deltaRGBs[...,2]))/2)

    r_a = sumR*r_gain + cnt_offset + ctt_r
    r_b = g_offset - cnt_offset/2 #sumR2/2 # + np.amax(deltaRGBs[...,1])
    r_c = b_offset - cnt_offset/2 # + np.amax(deltaRGBs[...,2])

    g_b = sumG*g_gain  + cnt_offset + ctt_g
    g_a = r_offset - cnt_offset/2  # + np.amax(deltaRGBs[...,0])
    g_c = b_offset - cnt_offset/2 # + np.amax(deltaRGBs[...,2])

    b_c = sumB*b_gain + cnt_offset  + ctt_b
    b_a = r_offset - cnt_offset/2 # + np.amax(deltaRGBs[...,0])
    b_b = g_offset - cnt_offset/2 # + np.amax(deltaRGBs[...,1])


    ccm = [r_a, r_b, r_c, g_a, g_b, g_c, b_a, b_b, b_c]

    return ccm


FILE_PATH = ".\img" #os.path.join('/', 'Users', 'ats', 'hobbies', 'notebooks', 'assets')
ILUMINATIONS = ['daylightLow', ]
DEVICES = ['img']
IMGS = {}
#=======================================================================#

for ilumination in ILUMINATIONS:
    IMGS[ilumination] = {}
    for device in DEVICES:
        filepath = GOPR_FILE_NAME
        #print(filepath)
        assert os.path.isfile(GOPR_FILE_NAME)
        img = Image.open(filepath).rotate(0) # Rotate to position the WHITE on the bottom left corner
        IMGS[ilumination][device] = crop_center(img, IMG_SIZE[0], IMG_SIZE[1])


for ilumination in ILUMINATIONS:
    #print(ilumination)
    comp_img = np.zeros((IMG_SIZE[1], IMG_SIZE[0]*len(DEVICES),3), 'uint8')
    for i, device in enumerate(DEVICES):
        img = np.array(IMGS[ilumination][device])
        comp_img[:, IMG_SIZE[0]*i:IMG_SIZE[0]*(i+1), :] = img
    #Image.fromarray(comp_img).show()

CHARTS = {}
CHECKER_SRCS = {}
for ilumination in ILUMINATIONS:
    CHARTS[ilumination] = {}
    CHECKER_SRCS[ilumination] = {}
    #print(ilumination)
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


print(src_rgb)
labels = [
    'dark skin', 'light skin', 'blue sky', 'foliage', 'blue flower', 'bluish green',
    'orange', 'purplish blue', 'moderate red', 'purple', 'yellow green', 'orange yellow',
    'blue', 'green', 'red', 'yellow', 'magenta', 'cyan',
    'white', 'neutral 8', 'neutral 6.5', 'neutral 5', 'neutral 3.5', 'black'
]

color_checker_rgb = np.zeros((24, 3))
for j in range(24):
    color_checker_rgb[j] = [src_rgb[j][0]/255,src_rgb[j][1]/255,src_rgb[j][2]/255]
clrs = plot_rgb_chart(color_checker_rgb)
Image.fromarray(plot_rgb_chart(color_checker_rgb, (50, 50))).show()

#======================================================================#

for ilumination in ILUMINATIONS:
    IMGS[ilumination] = {}
    for device in DEVICES:
        filepath = IMG_FILE_NAME
        #print(filepath)
        assert os.path.isfile(IMG_FILE_NAME)
        img = Image.open(filepath).rotate(0) # Rotate to position the WHITE on the bottom left corner
        IMGS[ilumination][device] = crop_center(img, IMG_SIZE[0], IMG_SIZE[1])


for ilumination in ILUMINATIONS:
    #print(ilumination)
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
    #print(ilumination)
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
    sRGB_Color = np.zeros((24, 3))
    for j in range(24):
        sRGB_Color[j] = color_checker_rgb[j]
    Image.fromarray(plot_rgblx2(sRGB_Color,src,(50, 50))).show()
"""
rgb_table = np.zeros((24, 3))
for j in range(24):
    rgb_table[j] = [color_checker_rgb[j][0]/255,color_checker_rgb[j][1]/255,color_checker_rgb[j][2]/255]
clrs = plot_rgb_chart(rgb_table)
Image.fromarray(plot_rgb_chart(rgb_table, (50, 50))).show()
"""
GOPRO_INDEX = 0
IMX_INDEX = 1

normalized_rgbs = np.array(srcs)
rows = len(normalized_rgbs)
columns = len(normalized_rgbs[GOPRO_INDEX])

assert rows == 1
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
    #print(normalized_rgbs[0][j][0])
    chack_in_A = convert_color(sRGBColor(color_checker_rgb[j][0]/255,color_checker_rgb[j][1]/255,color_checker_rgb[j][2]/255), LabColor)
    chack_in_B = convert_color(sRGBColor(normalized_rgbs[0][j][0][0],normalized_rgbs[0][j][0][1],normalized_rgbs[0][j][0][2]), LabColor)

    deltE = delta_e_cie2000(chack_in_A,chack_in_B )
    print(f'{labels[j]}: {deltE}')
    deltEs[j] = deltE
# Plot colors
print(f'Total: {deltEs.sum()}')

deltaRGBs = np.zeros((24, 3))
rgb_table = np.zeros((24, 3))
for j in range(24):
    chack_in_A = convert_color(sRGBColor(color_checker_rgb[j][0],color_checker_rgb[j][1],color_checker_rgb[j][2]), sRGBColor)
    chack_in_B = convert_color(sRGBColor(normalized_rgbs[0][j][0][0],normalized_rgbs[0][j][0][1],normalized_rgbs[0][j][0][2]), sRGBColor)

    deltaRGB = [chack_in_A.rgb_r - chack_in_B.rgb_r,
                        chack_in_A.rgb_g - chack_in_B.rgb_g,
                        chack_in_A.rgb_b - chack_in_B.rgb_b]
    deltaRGBs[j] = [deltaRGB[0],deltaRGB[1],deltaRGB[2]]

    #tmp = convert_color(deltaRGB, sRGBColor, target_illuminant='d65')
    #rgb_table[j] = [tmp.rgb_r,tmp.rgb_g,tmp.rgb_b]
#clrs = plot_rgb_chart(rgb_table)
Image.fromarray(plot_rgb_chart(deltaRGBs, (50, 50))).show()
#print(deltaRGBs)

for j in range(24):
    chack_in_A = convert_color(sRGBColor(color_checker_rgb[j][0]/255,color_checker_rgb[j][1]/255,color_checker_rgb[j][2]/255), sRGBColor)
    chack_in_B = convert_color(sRGBColor(normalized_rgbs[0][j][0][0],normalized_rgbs[0][j][0][1],normalized_rgbs[0][j][0][2]), sRGBColor)

    chack_in_B.rgb_r = chack_in_B.rgb_r + deltaRGBs[j][0]
    chack_in_B.rgb_g = chack_in_B.rgb_g + deltaRGBs[j][1]
    chack_in_B.rgb_b = chack_in_B.rgb_b + deltaRGBs[j][2]

    tmp = convert_color(chack_in_B, sRGBColor)
    rgb_table[j] = [chack_in_B.rgb_r,chack_in_B.rgb_g,chack_in_B.rgb_b]

    deltE = delta_e_cie1976( convert_color(chack_in_A,LabColor),
                             convert_color(chack_in_B,LabColor)
                            )
    print(f'{labels[j]}: {deltE}')
    deltEs[j] = deltE
print(f'Total: {deltEs.sum()}')


print(deltaRGBs)


ccm = creat_ccm_2(deltaRGBs)

print("-------------------------------------------")
print( "%0.8f , %0.8f , %0.8f ,"  % (ccm[0],ccm[1],ccm[2]))
print( "%0.8f , %0.8f , %0.8f ,"  % (ccm[3],ccm[4],ccm[5]))
print( "%0.8f , %0.8f , %0.8f "  % (ccm[6],ccm[7],ccm[8]))






