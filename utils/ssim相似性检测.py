import glob
import os

from cv2 import cv2
from skimage.metrics import structural_similarity

real_path = r'D:\Dataset\data\final\train\outdoor'
real_path = r'D:\Dataset\data\final-8\train\outdoor'
# real_path = r'../../final/train/outdoor'

if __name__ == '__main__':
    width = 224
    height = 224
    real_name_lists = glob.glob(os.path.join(real_path, '*.jpg'))
    for idx, img_1 in enumerate(real_name_lists):  # 非顺序，不过都是pair
        # if 'sunny' not in img_1:
        #     continue
        real_img = cv2.imread(img_1)
        fake_img = cv2.imread(r'./src.jpg')

        real_img = cv2.resize(real_img, (width, height))
        fake_img = cv2.resize(fake_img, (width, height))

        ssim = structural_similarity(fake_img, real_img, multichannel=True)  # 对于多通道图像(RGB、HSV等)关键词multichannel要设置为True
        if ssim > 0.5:
            print('img_1:{}, SSIM：{}'.format(str(img_1), ssim))
