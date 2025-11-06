import cv2
from typing import List, Tuple
from pathlib import Path
import yaml
import os

PAD_UNION_PX = 50     # padding around each cluster union box
MIN_CC_AREA  = 500    # min area for refined component
KERNEL_SZ    = 5      # morphology closing kernel
CLOSE_ITERS  = 7      # closing iterations
L_ALPHA = 1.25
L_BETA = -50
L_CHANNEL = 0
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
MORPH_ITER = 3
LARGE_AREA_THRESHOLD = 10000
MIN_AREA = 200
MAX_WIDTH = 1000
MAX_HEIGHT = 1000
ROI_PAD = 50
MIN_DISPLAY_SIZE = 384
TARGET_DPI = 1600

M1_LABELS: List[str] = [
    "Cixiidae","Planthopper","Beetle","Aphid",
    "Fly","Thrips","Hymenopteran","Mosquito",
    "Spider","Ants","Lacewings","Other Insects","Artefact"
]

M2_LABELS: List[str] = ['Planthopper', 'Pentastiridius', 'Hyalesthes', 'Reptalus']

if Path.cwd().stem != 'src':
    CONFIG_PATH = Path('src/config.yaml').resolve()
else:
    CONFIG_PATH = Path('config.yaml').resolve()
    
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)


MODEL_PATHS = {
    "M1" : (config['M1_MODEL_PATH'][0], config['M1_MODEL_PATH'][1], 8192, 600, 13),
    "M2" : (config['M2_MODEL_PATH'][0], config['M1_MODEL_PATH'][1], 8192, 600, 4)
}
