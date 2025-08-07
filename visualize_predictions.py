import argparse
import logging
import os
import random
import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from pycocotools.coco import COCO
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.checkpoint import DetectionCheckpointer

from detectron2.modeling.meta_arch import MMEhoiNetv1
from detectron2.utils.converters import MMEhoiNetConverterv1
from detectron2.utils.custom_visualizer import EhoiVisualizerv1

from detectron2.modeling.meta_arch.MiDaS import utils as midas_utils
from torchvision.transforms import Compose
from detectron2.modeling.meta_arch.MiDaS.midas.transforms import Resize, PrepareForNet

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

def draw_text_with_outline(img, text, pos, font_face=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.6, text_color=(255, 255, 255), thickness=2):
    """Disegna testo con un contorno per una migliore leggibilità."""
    x, y = pos
    cv2.putText(img, text, (x, y), font_face, font_scale, (0, 0, 0), thickness * 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font_face, font_scale, text_color, thickness, cv2.LINE_AA)
    return img

def draw_ground_truth_custom(image, annotations, metadata):
    result = image.copy()
    overlay = image.copy()
    class_names = metadata.thing_classes
    
    hand_category_id = -1
    for k, v in metadata.thing_dataset_id_to_contiguous_id.items():
        if v == metadata.hand_id:
            hand_category_id = k
            break
            
    hand_anns = [ann for ann in annotations if ann.get('category_id') == hand_category_id]
    anns_by_id = {ann.get('id'): ann for ann in annotations}

    for i, ann in enumerate(annotations):
        if 'segmentation' in ann and ann['segmentation']:
            contiguous_id = metadata.thing_dataset_id_to_contiguous_id.get(ann['category_id'], 0)
            hue = int((contiguous_id * 23 + i * 17) % 180)
            color = tuple(map(int, cv2.cvtColor(np.uint8([[[hue, 200, 220]]]), cv2.COLOR_HSV2BGR)[0][0]))
            for poly in ann['segmentation']:
                if isinstance(poly, list) and len(poly) >= 6:
                    points = np.array(poly).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(overlay, [points], color)

    cv2.addWeighted(overlay, 0.4, result, 0.6, 0, result)

    contacting_object_ids = {ann['id_obj'] for ann in hand_anns if ann.get('contact_state') == 1 and 'id_obj' in ann}
    for ann in annotations:
        if 'bbox' not in ann or len(ann['bbox']) != 4: continue
        x, y, w, h = [int(c) for c in ann['bbox']]
        cat_id = ann.get('category_id')
        contiguous_id = metadata.thing_dataset_id_to_contiguous_id.get(cat_id, -1)
        
        color, label = (None, None)
        if contiguous_id == metadata.hand_id:
            contact_state = ann.get('contact_state', -1)
            side_text = "L Hand" if ann.get('hand_side') == 0 else "R Hand"
            label = f"{side_text}{' CONTACT' if contact_state == 1 else ''}"
            color = (0, 255, 0) if contact_state == 1 else (0, 0, 255)
        else:
            label = class_names[contiguous_id] if contiguous_id != -1 else f"ID:{cat_id}"
            is_in_contact = ann.get('id') in contacting_object_ids
            color = (0, 255, 0) if is_in_contact else (255, 100, 0)
        
        cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
        draw_text_with_outline(result, label, (x, y - 10), text_color=color)

    for ann in hand_anns:
        if ann.get('contact_state') == 1 and 'id_obj' in ann and ann['id_obj'] in anns_by_id:
            obj_ann = anns_by_id.get(ann['id_obj'])
            if obj_ann and 'bbox' in obj_ann:
                hx, hy, hw, hh = [int(c) for c in ann['bbox']]
                ox, oy, ow, oh = [int(c) for c in obj_ann['bbox']]
                hand_center, obj_center = (hx + hw // 2, hy + hh // 2), (ox + ow // 2, oy + oh // 2)
                cv2.arrowedLine(result, hand_center, obj_center, (0, 0, 255), 2, tipLength=0.3)
        if 'keypoints' in ann and ann['keypoints']:
            kps = np.array(ann['keypoints']).reshape(-1, 3)
            for kx, ky, v in kps:
                if v > 0: cv2.circle(result, (int(kx), int(ky)), 4, (255, 255, 0), -1)
    return result

def create_comparison_visualization(gt_image, pred_image):
    h1, w1 = gt_image.shape[:2]
    h2, w2 = pred_image.shape[:2]
    if h1 != h2:
        target_h = max(h1, h2)
        gt_image = cv2.resize(gt_image, (int(w1 * target_h / h1), target_h))
        pred_image = cv2.resize(pred_image, (int(w2 * target_h / h2), target_h))

    gt_pil = Image.fromarray(cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB))
    pred_pil = Image.fromarray(cv2.cvtColor(pred_image, cv2.COLOR_BGR2RGB))
    
    padding, header_h = 20, 80
    total_w = gt_pil.width + pred_pil.width + padding * 3
    total_h = gt_pil.height + header_h + padding * 2
    
    canvas = Image.new('RGB', (total_w, total_h), (240, 240, 240))
    draw = ImageDraw.Draw(canvas)
    try: font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=40)
    except IOError: font = ImageFont.load_default()
        
    canvas.paste(gt_pil, (padding, padding + header_h))
    canvas.paste(pred_pil, (gt_pil.width + padding * 2, padding + header_h))
    
    for i, text in enumerate(["Ground Truth", "Prediction"]):
        img = gt_pil if i == 0 else pred_pil
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        start_x = padding if i == 0 else gt_pil.width + padding * 2
        text_x = start_x + (img.width - text_w) // 2
        text_y = padding + (header_h - text_h) // 2
        draw.text((text_x, text_y), text, font=font, fill=(50, 50, 50))
        
    return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)

def find_latest_training_dir(base_dir="./output_dir"):
    all_subdirs = [d for d in Path(base_dir).iterdir() if d.is_dir() and d.name != 'last_training']
    return max(all_subdirs, key=lambda d: d.stat().st_mtime) if all_subdirs else None

def find_best_weights_file(training_dir: Path):
    final_model = training_dir / "model_final.pth"
    if final_model.exists(): return final_model
    checkpoints = sorted(training_dir.glob("model_*.pth"))
    return max(checkpoints, default=None) if checkpoints else None

def main(args):
    training_dir = find_latest_training_dir(args.training_dir.rsplit('/', 1)[0]) or Path(args.training_dir)
    if not training_dir or not training_dir.exists():
        logger.error(f"Training directory not found: {args.training_dir}")
        return
    logger.info(f"Using artifacts from training directory: {training_dir}")
    
    config_file = training_dir / "cfg.yaml"
    weights_file = find_best_weights_file(training_dir)

    if not config_file.exists() or not weights_file:
        logger.error(f"Config or weights file not found in {training_dir}")
        return

    dataset_name = "ehoi_visualizer_dataset"
    if dataset_name in DatasetCatalog.list():
        DatasetCatalog.remove(dataset_name)
    image_root = str(Path(args.input_json).parent.parent / "images")
    register_coco_instances(dataset_name, {}, args.input_json, image_root)
    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)

    cfg = get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = str(weights_file)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.conf_threshold
    cfg.freeze()

    try:
        hand_class_name = next(name for name in ["hand", "mano"] if name in metadata.thing_classes)
        metadata.hand_id = metadata.thing_classes.index(hand_class_name)
    except (StopIteration, ValueError):
        logger.error(f"Could not find 'hand' or 'mano' in dataset categories: {metadata.thing_classes}")
        return
    
    model = MMEhoiNetv1(cfg, metadata)
    model.eval()
    model.to("cuda")
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    converter = MMEhoiNetConverterv1(cfg, metadata)
    custom_visualizer = EhoiVisualizerv1(cfg, metadata, converter)
    
    coco_api = COCO(args.input_json)
    os.makedirs(args.output_dir, exist_ok=True)
    
    transform_for_depth = Compose([
        Resize(
            cfg.ADDITIONAL_MODULES.DEPTH_MODULE.NET_W,
            cfg.ADDITIONAL_MODULES.DEPTH_MODULE.NET_H,
            resize_target=True, keep_aspect_ratio=True, ensure_multiple_of=32,
            resize_method=cfg.ADDITIONAL_MODULES.DEPTH_MODULE.RESIZE_MODE,
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        PrepareForNet(),
    ])
    
    num_to_visualize = min(args.num_samples, len(dataset_dicts))
    #samples_to_process = random.sample(dataset_dicts, num_to_visualize)
    samples_to_process = dataset_dicts[:num_to_visualize]
    
    for i, image_dict in enumerate(samples_to_process):
        file_name = os.path.basename(image_dict["file_name"])
        logger.info(f"Processing sample {i+1}/{num_to_visualize}: {file_name}")
        
        img = cv2.imread(image_dict["file_name"])
        if img is None: continue

        height, width = img.shape[:2]
        image_tensor = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
        
        img_for_midas = midas_utils.read_image(image_dict["file_name"])
        image_for_depth_module_tensor = transform_for_depth({"image": img_for_midas})["image"]
        
        inputs = [{
            "image": image_tensor, 
            "height": height, 
            "width": width, 
            "image_for_depth_module": image_for_depth_module_tensor
        }]
        
        with torch.no_grad():
            prediction = model(inputs)[0]
            
        gt_annotations = coco_api.loadAnns(coco_api.getAnnIds(imgIds=image_dict['image_id']))
        gt_image = draw_ground_truth_custom(img.copy(), gt_annotations, metadata)
        pred_image = custom_visualizer.draw_results(img.copy(), prediction)
            
        comparison_image = create_comparison_visualization(gt_image, pred_image)
        cv2.imwrite(os.path.join(args.output_dir, f"comparison_{file_name}"), comparison_image)

    logger.info(f"Visualizations successfully saved to: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize model predictions against ground truth for the EHOI project.")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of random samples to visualize.")
    parser.add_argument("--input-json", type=str, default="data/egoism-hoi-dataset/annotations/val_coco.json", help="Path to the COCO JSON annotation file.")
    parser.add_argument("--output-dir", default="./visualization_output_kpt", help="Directory to save the visualization images.")
    parser.add_argument("--conf-threshold", type=float, default=0.5, help="Confidence threshold for displaying predictions.")
    parser.add_argument("--training-dir", type=str, default="output_dir/last_training", help="Path to the training directory for loading artifacts.")
    
    args = parser.parse_args()
    main(args)