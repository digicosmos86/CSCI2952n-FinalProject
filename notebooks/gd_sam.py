import numpy as np
import torch
import torchvision

from segment_anything import SamPredictor
from groundingdino.util.inference import predict

from sam import make_sam_model
from gd import make_gd_model

gd_model = make_gd_model()
sam_model = make_sam_model()

sam_predictor = SamPredictor(sam_model)

device="cuda"

def bboxes_transform(bboxes, image):
    """Transforms the output bounding boxes of GroundingDINO
    in scaled cxcywh format to SAM's xyxy in image coordinates.
    
    The input image is in RGB format with dimensions (h, w, 3)
    """
    
    h, w, _ = image.shape
    image_shape = torch.tensor([w, h, w, h])
    bboxes = torchvision.ops.box_convert(boxes=bboxes, in_fmt="cxcywh", out_fmt="xyxy")
    
    bboxes = (bboxes * image_shape).numpy()
    batch_boxes = sam_predictor.transform.apply_boxes(bboxes, image.shape[:2])
    return torch.tensor(batch_boxes, device=device)


def apply_boxes(bboxes, image):
    batch_boxes = bboxes_transform(bboxes, image)
    
    sam_predictor.set_image(image)
    masks, scores, _ = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=batch_boxes,
        multimask_output=False,
    )
    
    return masks, scores


def gd_sam_predict(
    gd_image,
    sam_image,
    text_prompt="small ice chunks , tiny white caps",
    score_threshold=0.9,
    box_threshold=0.07,
    brightness_threshold=0.5
):
    
    boxes, logits, phrases = predict(
        model=gd_model, 
        image=gd_image, 
        caption=text_prompt, 
        box_threshold=box_threshold,
        text_threshold=box_threshold
    )
    
    if len(boxes) <= 20:
        return None
    
    filtered_results = filter_boxes(boxes, logits, phrases, sam_image)
    
    if filtered_results is None or len(filtered_results[0]) <= 20:
        return None
    
    boxes, logits, phrases = filtered_results
    
    masks, scores = apply_boxes(boxes, sam_image)
    
    masks = masks[scores >= score_threshold].cpu()
    
    ## Filter dark patches
    filter_masks = np.stack([
        sam_image[mask.squeeze()].mean() > 255. * brightness_threshold
        for mask in masks if mask.sum() < 50000
    ])
    
    return masks[filter_masks, :, :]

def filter_boxes(boxes, logits, phrases, sam_image, max_width=260, max_height=260):
    if len(boxes) == 0:
        return None
    
    h, w, _ = sam_image.shape
    
    keep_idx = (boxes[:, 2] <= max_width / w) & (boxes[:, 3] <= max_height / h)
    
    if len(keep_idx) == 0:
        return None
    
    return boxes[keep_idx, :], logits[keep_idx], np.array(phrases)[keep_idx].tolist()