'''
    Este script recebe de entrada o nome do arquivo .csv onde serão salvas as
    características extraídas e o nome das pastas que contém as imagens de
    cada classe

    Exemplo:

    "python classificador.py sinais A B C D E"

    Extrai as características das imagens nas pastas A, B, C, D e E e salva os
    dados no arquivo sinais.csv
'''

import cv2
import sys
import csv
import numpy as np
from glob import glob


def hsv_thresh(img):
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    l1 = np.array((5, 38, 30))
    l2 = np.array((50, 250, 242))
    skinMask = cv2.inRange(frame, l1, l2)
    return skinMask


folders = sys.argv[2:]
with open(sys.argv[1] + '.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    columnTitle = ["feat1", "feat2", "feat3", "feat4", "feat5", "feat6", "feat7", "class"]
    writer.writerow(columnTitle)

    for folder in folders:
        print("Folder " + folder)
        for file in glob(folder + "/*"):
            img = cv2.imread(file)
            name = file.split('/')[1]
            img = hsv_thresh(img)

            cv2.imwrite("segmentation/" + name, img)

            hu = cv2.HuMoments(cv2.moments(img)).flatten()
            hu = np.array(hu).tolist()
            hu.append(folder)
            writer.writerow(hu)
