# io_utils.py
from PIL import Image
import numpy as np
import hashlib
from io import BytesIO

def read_image_pil(uploaded):
    """Return (rgb_uint8_array, dpi_or_None, raw_bytes, file_hash). Accepts a file-like object or bytes."""
    raw = uploaded.read() if hasattr(uploaded, "read") else uploaded
    img = Image.open(BytesIO(raw)).convert("RGB")
    dpi_meta = img.info.get("dpi")  # often (xdpi, ydpi) or a single int
    if isinstance(dpi_meta, (tuple, list)):
        dpi = int(dpi_meta[0]) if dpi_meta and dpi_meta[0] else None
    else:
        dpi = int(dpi_meta) if dpi_meta else None

    file_hash = hashlib.sha1(raw).hexdigest()
    
    return np.asarray(img, dtype=np.uint8), dpi, file_hash
