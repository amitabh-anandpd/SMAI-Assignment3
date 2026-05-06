import os
os.environ["HUB_DATASET_ENDPOINT"] = "https://modelscope.cn/api/v1/datasets"
# Try disabling pir
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
import cv2
import numpy as np
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=False, lang="en", device="cpu")
img = np.zeros((100, 200, 3), dtype=np.uint8)
res = ocr.ocr(img)
print(res)
