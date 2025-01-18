import cv2
import numpy as np
from PIL import Image


def read_image(image_path, img_type="RGB"):
    """从 image_path 从读取图片
    如果 img_type="RGB"，则直接读取；
    如果 img_type="BGR"，则将 BGR 转换为 RGB
    """
    if img_type == "RGB":
        return Image.open(image_path).convert('RGB')
    elif img_type == "BGR":
        img = Image.open(image_path).convert('RGB')
        img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        return img
    else:
        raise NotImplementedError
