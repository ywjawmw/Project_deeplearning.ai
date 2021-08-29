import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
# from scipy.misc import imresize

# root path depends on your computer
root = '/home/ywj/Myproject/Project_deeplearning.ai/experience/数据集/图像生成CelebA/Img/img_align_celeba/'
save_root = '/home/ywj/Myproject/Project_deeplearning.ai/experience/数据集/图像生成CelebA/Img/resized_celebA/'
resize_size = 64

if not os.path.isdir(save_root):
    os.mkdir(save_root)
if not os.path.isdir(save_root + 'celebA'):
    os.mkdir(save_root + 'celebA')
img_list = os.listdir(root)

# ten_percent = len(img_list) // 10

for i in range(len(img_list)):
    img = plt.imread(root + img_list[i])
    img = np.array(Image.fromarray(img).resize((resize_size, resize_size)))
    plt.imsave(fname=save_root + 'celebA/' + img_list[i], arr=img)

    if (i % 1000) == 0:
        print('%d images complete' % i)