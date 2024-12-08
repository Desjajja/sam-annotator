import gradio as gr
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw
from segment_anything.onnxpredictor import SamOnnxPredictor
import cv2
import logging
import hashlib
from pathlib import Path
from functools import reduce
import os

logging.basicConfig(level=logging.INFO)
# Path to your ONNX model
VIT_PATH = "./models/vit_quantized.onnx"
MASK_PATH = "./models/sam_vit_h.onnx"

# Load the ONNX model
# vit_session = ort.InferenceSession(VIT_PATH)
mask_session = ort.InferenceSession(MASK_PATH)
predictor = SamOnnxPredictor(VIT_PATH)
# image_embedding = gr.State(None)
image_loaded = None
embedding_loaded = None
image_homepath = None
mask_list = []

def undo_masking():
    global mask_list
    mask_list.pop()
    return draw_output()

def clear_masks():
    global mask_list
    mask_list = []
    return draw_output()

def load_image(image_path: str) -> np.array:
    """
    Perform segmentation using the ONNX model.
    """
    logging.info(f"loading image: {image_path}")
    global image_loaded
    image_loaded = cv2.imread(image_path)
    hash = hashlib.md5(image_loaded).hexdigest()
    
    global mask_list
    mask_list = []

    
    global image_homepath
    image_homepath = Path(image_path).parent
    embed_path = image_homepath / f"{hash}.npy"
    
    global embedding_loaded
    if os.path.exists(embed_path):
        embedding_loaded = np.load(embed_path)
    else:
        # image_embedding = load_image(image_loaded)
        hash = hashlib.md5(image_loaded.tobytes()).hexdigest()
        image_cvt = cv2.cvtColor(image_loaded, cv2.COLOR_BGR2RGB)
        logging.debug(f"type of image: {type(image_cvt)}")
        predictor.set_image(image_cvt)
        embedding_loaded = predictor.get_image_embedding()[0]
        np.save(image_homepath / f"{hash}.npy", embedding_loaded)
    
    return draw_output()

def segment_image(coord: tuple[int, int] = None):
    """
    Perform segmentation using the ONNX model.
    """
    # global image_loaded
    # image_loaded = cv2.imread(image_path)
    # hash = hashlib.md5(image_loaded).hexdigest()
    # global image_homepath
    # image_homepath = Path(image_path).parent
    # embed_path = image_homepath / f"{hash}.npy"
    # if os.path.exists(embed_path):
    #     image_embedding = np.load(embed_path)
    # else:
    #     image_embedding = load_image(image_loaded)
    # logging.debug(f"image embedding: {image_embedding}")
    input_point = np.array([coord])
    input_label = np.array([1])
    
    onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
    onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)

    global image_loaded
    onnx_coord = predictor.transform.apply_coords(onnx_coord, image_loaded.shape[:2]).astype(np.float32)

    onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    onnx_has_mask_input = np.zeros(1, dtype=np.float32)
    
    global embedding_loaded
    ort_inputs = {
    "image_embeddings": embedding_loaded,
    "point_coords": onnx_coord,
    "point_labels": onnx_label,
    "mask_input": onnx_mask_input,
    "has_mask_input": onnx_has_mask_input,
    "orig_im_size": np.array(image_loaded.shape[:2], dtype=np.float32)
    }
    
    masks, _, low_res_logits = mask_session.run(None, ort_inputs)
    masks = masks > predictor.model.mask_threshold
    global mask_list
    if mask_list:
        mask_list.append(masks[:,0,:,:])
    else:
        mask_list = [masks[:,0,:,:]]
    return draw_output()
    
    
def draw_output():
    global mask_list
    global image_loaded
    mask = reduce(np.logical_or, mask_list) if len(mask_list) else None
    logging.info(f"drawing output: mask: {mask}")
    image_cvt = cv2.cvtColor(image_loaded, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_cvt.astype('uint8')).convert("RGBA")
    if mask is not None:
        mask_image = draw_mask(mask)
        bbox = compute_bbox(mask)
        output_img = Image.alpha_composite(
            image_pil,
            Image.fromarray((mask_image * 255).astype('uint8'), mode="RGBA"))
        # draw bbox over the output img
        draw = ImageDraw.Draw(output_img)
        draw.rectangle(bbox, outline=(255, 0, 0), width=5)
    # logging.info(mask_image)
    else:
        output_img = image_pil
    return output_img

def get_select_coords(evt: gr.SelectData):
    return segment_image(evt.index)

def draw_mask(mask):
    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    return mask_image

def compute_bbox(mask):
    logging.info(f"image shape: {image_loaded.shape}")
    logging.info(f"mask shape: {mask.shape}")
    mask_pos = np.argwhere(mask.squeeze(0))
    logging.info(f"mask pos: {mask_pos}")
    bbox = [
        int(np.min(mask_pos[:, 1])),
        int(np.min(mask_pos[:, 0])),
        int(np.max(mask_pos[:, 1])),
        int(np.max(mask_pos[:, 0]))
    ]
    logging.info(f"bbox: {bbox}")
    return bbox

with gr.Blocks() as demo:
    gr.Markdown("""
    # SAM ONNX Annotator
    """)
    with gr.Row():
        input_img = gr.Image(type="filepath")
        output_img = gr.Image(type="pil")
    
    input_img.upload(load_image, [input_img], [output_img])
    input_img.change(load_image, [input_img], [output_img])
    input_img.select(get_select_coords, [], [output_img])

    
    with gr.Row():
        rst_bbox = gr.Textbox(
            label="Result",
            info="(x1, y1), (x2, y2)",
            lines=2,
        )
    
    with gr.Row():
        btn_undo = gr.Button("Undo")
        btn_clear = gr.ClearButton() 
    btn_undo.click(undo_masking, [], [output_img])
    btn_clear.click(clear_masks, [], [output_img])
    
    
if __name__ == "__main__":
    demo.launch()
