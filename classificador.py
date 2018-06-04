'''
    Este script recebe de entrada o nome do arquivo .csv onde serão salvas as
    características extraídas e o nome das pastas que contém as imagens de
    cada classe

    Exemplo:

    "python classificador.py sinais A B C D E"

    Extrai as características das imagens nas pastas A, B, C, D e E e salva os
    dados no arquivo sinais.csv
'''

import sys
import csv
from glob import glob
from extraction import extract_features

def main():
    folders = sys.argv[2:]
    with open(sys.argv[1] + '.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        columnTitle = ["feat1", "feat2", "feat3", "feat4", "feat5", "feat6", "feat7", "class"]
        writer.writerow(columnTitle)

        for folder in folders:
            print("Folder " + folder)
            for file in glob(folder + "/*"):
                hu = extract_features(file)
                hu.append(folder)
                writer.writerow(hu)

if __name__ == "__main__":
    main()
