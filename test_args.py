import os
os.environ.setdefault("HUB_DATASET_ENDPOINT", "https://modelscope.cn/api/v1/datasets")
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_use_onednn"] = "0"

from paddleocr import PaddleOCR

ocr = PaddleOCR(lang="en", device="cpu", use_angle_cls=False, use_doc_preprocessor=False, use_doc_orientation_classify=False, use_textline_orientation=False)
print("Initialized successfully!")
