from cv2 import imread, cvtColor, inRange, HuMoments, moments, COLOR_BGR2HSV
from numpy import array

def extract_features(file):
    img = imread(file)
    frame = cvtColor(img, COLOR_BGR2HSV)
    l1 = array((5, 38, 30))
    l2 = array((50, 250, 242))
    skin_mask = inRange(frame, l1, l2)
    hu = HuMoments(moments(skin_mask)).flatten()
    hu = array(hu).tolist()
    return hu

if __name__ == "__main__":
    main()
