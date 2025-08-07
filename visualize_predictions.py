import argparse
import logging
import os
import random
import cv2
import torch
import numpy as np
import json
from pathlib import Path

# NUOVA DIPENDENZA per la grafica
from PIL import Image, ImageDraw, ImageFont

from pycocotools.coco import COCO
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

from detectron2.modeling.meta_arch import MMEhoiNetv1

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# --- TUA FUNZIONE DI DISEGNO (invariata, è già perfetta) ---
def draw_annotations_custom(image, annotations, metadata):
    # ... (il tuo codice qui, esattamente come prima) ...
    result = image.copy()
    overlay = image.copy()
    
    class_names = {i: name for i, name in enumerate(metadata.thing_classes)}
    hand_category_id = metadata.hand_id

    hand_anns = [ann for ann in annotations if ann.get('category_id') == hand_category_id]
    other_anns = [ann for ann in annotations if ann.get('category_id') != hand_category_id]
    
    anns_by_id = {ann.get('id'): ann for ann in annotations}

    segmentation_alpha = 0.5
    for i, ann in enumerate(annotations):
        if 'segmentation' in ann and ann['segmentation']:
            hue = int((ann.get('id', 0) * 17) % 180)
            color = tuple(map(int, cv2.cvtColor(np.uint8([[[hue, 255, 200]]]), cv2.COLOR_HSV2BGR)[0][0]))
            for poly in ann['segmentation']:
                if isinstance(poly, list) and len(poly) >= 6:
                    points = np.array(poly).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(overlay, [points], color)
    cv2.addWeighted(overlay, segmentation_alpha, result, 1 - segmentation_alpha, 0, result)

    contacting_object_ids = {ann['id_obj'] for ann in hand_anns if ann.get('contact_state') == 1 and 'id_obj' in ann}

    text_font = cv2.FONT_HERSHEY_SIMPLEX
    text_scale = 0.5
    text_thickness = 1

    for ann in other_anns:
        if 'bbox' in ann and len(ann['bbox']) == 4:
            x, y, w, h = [int(c) for c in ann['bbox']]
            cat_id = ann.get('category_id')
            label = class_names.get(cat_id, f"ID:{cat_id}")
            is_in_contact = ann.get('id') in contacting_object_ids
            color = (0, 255, 0) if is_in_contact else (255, 100, 0)
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
            cv2.putText(result, label, (x, y - 10), text_font, text_scale, (255, 255, 255), text_thickness)
    
    for ann in hand_anns:
        if 'bbox' in ann and len(ann['bbox']) == 4:
            x, y, w, h = [int(c) for c in ann['bbox']]
            contact_state = ann.get('contact_state', -1)
            side_text = "L Hand" if ann.get('hand_side') == 0 else "R Hand"
            label = f"{side_text}"
            color = (0, 255, 0) if contact_state == 1 else (0, 0, 255)
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
            cv2.putText(result, label, (x, y - 10), text_font, text_scale, (255, 255, 255), text_thickness)

    for ann in hand_anns:
        if ann.get('contact_state') == 1 and 'id_obj' in ann and ann.get('id_obj') in anns_by_id:
            obj_ann = anns_by_id[ann['id_obj']]
            hand_bbox = ann.get('bbox')
            obj_bbox = obj_ann.get('bbox')
            
            if hand_bbox and obj_bbox:
                hx, hy, hw, hh = [int(c) for c in hand_bbox]
                ox, oy, ow, oh = [int(c) for c in obj_bbox]
                hand_center = (hx + hw // 2, hy + hh // 2)
                obj_center = (ox + ow // 2, oy + oh // 2)
                cv2.arrowedLine(result, hand_center, obj_center, (0, 0, 255), 2, tipLength=0.3)

    for ann in hand_anns: 
        if 'keypoints' in ann and ann['keypoints']:
            kps = np.array(ann['keypoints']).reshape(-1, 3)
            for kx, ky, v in kps:
                if v > 0:
                    cv2.circle(result, (int(kx), int(ky)), 4, (255, 255, 0), -1)
    return result

# --- FUNZIONE DI DISEGNO PREDIZIONI (invariata) ---
def draw_predictions_custom(image, prediction, metadata, cfg):
    # ... (il tuo codice qui, esattamente come prima) ...
    result = image.copy()
    pred_instances = prediction["instances"].to("cpu")
    additional_outputs = prediction.get("additional_outputs")
    hand_instances = pred_instances[pred_instances.pred_classes == metadata.hand_id]

    pred_visualizer = Visualizer(result, metadata, scale=1.0)
    result = pred_visualizer.draw_instance_predictions(pred_instances).get_image()
    
    if additional_outputs and len(hand_instances) > 0 and additional_outputs.has("sides"):
        for i in range(len(additional_outputs.boxes)):
            box = additional_outputs.boxes[i].numpy().astype(int)
            x1, y1, x2, y2 = box
            
            side_code = additional_outputs.sides[i].item()
            side_text = "L Hand" if side_code == 0 else "R Hand"
            contact_state = additional_outputs.contact_states[i].item()
            contact_text = "CONTACT" if contact_state == 1 else "NO-CONTACT"
            score = additional_outputs.scores[i].item()
            label = f"{side_text}, {contact_text} ({score:.2f})"
            color = (0, 255, 0) if contact_state == 1 else (0, 0, 255)
            
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            cv2.putText(result, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if contact_state == 1 and additional_outputs.has("dxdymagn_hand"):
                hand_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                dx, dy, magn = additional_outputs.dxdymagn_hand[i].numpy()
                diag = np.sqrt(result.shape[0]**2 + result.shape[1]**2)
                scale_factor = cfg.ADDITIONAL_MODULES.ASSOCIATION_VECTOR_SCALE_FACTOR
                magnitude = (magn / scale_factor) * diag if scale_factor > 0 else magn * 100
                
                end_point = (int(hand_center[0] + dx * magnitude), int(hand_center[1] + dy * magnitude))
                cv2.arrowedLine(result, hand_center, end_point, (0, 0, 255), 2, tipLength=0.3)

    if hand_instances.has("pred_keypoints"):
        for kps in hand_instances.pred_keypoints:
             for x, y, v in kps:
                 if v > 0.1:
                     cv2.circle(result, (int(x), int(y)), 4, (255, 0, 0), -1)
    return result


# --- NUOVA FUNZIONE DI VISUALIZZAZIONE "PRO" CON PILLOW ---
def create_comparison_visualization(image_dict, prediction, metadata, cfg, coco_api):
    img = cv2.imread(image_dict["file_name"])
    if img is None:
        logger.error(f"Failed to read image: {image_dict['file_name']}")
        return None

    ann_ids = coco_api.getAnnIds(imgIds=image_dict['image_id'])
    annotations = coco_api.loadAnns(ann_ids)

    gt_image_cv = draw_annotations_custom(img.copy(), annotations, metadata)
    pred_image_cv = draw_predictions_custom(img.copy(), prediction, metadata, cfg)
    
    # Conversione da OpenCV (BGR) a Pillow (RGB)
    gt_image_pil = Image.fromarray(cv2.cvtColor(gt_image_cv, cv2.COLOR_BGR2RGB))
    pred_image_pil = Image.fromarray(cv2.cvtColor(pred_image_cv, cv2.COLOR_BGR2RGB))

    # Definizione del layout
    padding = 20
    header_height = 80
    background_color = (240, 240, 240) # Grigio chiaro
    font_color = (50, 50, 50) # Grigio scuro

    img_width, img_height = gt_image_pil.size
    total_width = (img_width * 2) + (padding * 3)
    total_height = img_height + header_height + (padding * 2)

    # Creazione della tela finale
    canvas = Image.new('RGB', (total_width, total_height), background_color)
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=40)
    except IOError:
        font = ImageFont.load_default()

    # Incolla le immagini sulla tela
    canvas.paste(gt_image_pil, (padding, padding + header_height))
    canvas.paste(pred_image_pil, (img_width + padding * 2, padding + header_height))

    # Disegna i titoli
    gt_text = "Ground Truth"
    pred_text = "Prediction"
    
    gt_text_bbox = draw.textbbox((0, 0), gt_text, font=font)
    pred_text_bbox = draw.textbbox((0, 0), pred_text, font=font)

    gt_text_x = padding + (img_width - (gt_text_bbox[2] - gt_text_bbox[0])) // 2
    gt_text_y = padding + (header_height - (gt_text_bbox[3] - gt_text_bbox[1])) // 2
    
    pred_text_x = (padding * 2) + img_width + (img_width - (pred_text_bbox[2] - pred_text_bbox[0])) // 2
    pred_text_y = padding + (header_height - (pred_text_bbox[3] - pred_text_bbox[1])) // 2

    draw.text((gt_text_x, gt_text_y), gt_text, font=font, fill=font_color)
    draw.text((pred_text_x, pred_text_y), pred_text, font=font, fill=font_color)
    
    # Riconverti in formato OpenCV per il salvataggio
    final_image_cv = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)
    return final_image_cv

def find_latest_training_dir(base_dir="./output_dir"):
    all_subdirs = [d for d in Path(base_dir).iterdir() if d.is_dir()]
    if not all_subdirs: return None
    return max(all_subdirs, key=lambda d: d.stat().st_mtime)

def find_best_weights_file(training_dir: Path):
    final_model = training_dir / "model_final.pth"
    if final_model.exists(): return final_model
    checkpoints = sorted(training_dir.glob("model_*.pth"))
    if checkpoints: return checkpoints[-1]
    return None

def main(args):
    training_dir = find_latest_training_dir()
    if not training_dir:
        logger.error("Could not find any training directory in './output_dir/'.")
        return
    logger.info(f"Using artifacts from latest training directory: {training_dir}")
    
    config_file = training_dir / "cfg.yaml"
    weights_file = find_best_weights_file(training_dir)

    if not config_file.exists() or not weights_file:
        logger.error(f"Config or weights file not found in {training_dir}")
        return

    cfg = get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(config_file)
    
    cfg.defrost()
    cfg.MODEL.WEIGHTS = str(weights_file)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.conf_threshold
    cfg.freeze()
    
    val_json_path = args.input_json
    if not val_json_path:
        default_path = Path("data/egoism-hoi-dataset/annotations/val_coco.json")
        if default_path.exists(): val_json_path = str(default_path)
        else:
            logger.error("Validation JSON not found. Please specify with --input-json.")
            return
    
    coco_api = COCO(val_json_path)

    dataset_name = "ehoi_visualizer_dataset"
    if dataset_name in DatasetCatalog.list(): DatasetCatalog.remove(dataset_name)
    
    image_root = str(Path(val_json_path).parent.parent / "images")
    register_coco_instances(dataset_name, {}, val_json_path, image_root)
    
    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)

    try:
        hand_class_name = "hand" if "hand" in metadata.thing_classes else "mano"
        metadata.hand_id = metadata.thing_classes.index(hand_class_name)
    except (AttributeError, ValueError):
        logger.error("Could not find 'hand' or 'mano' in dataset categories.")
        return

    predictor = DefaultPredictor(cfg)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    num_to_visualize = min(args.num_samples, len(dataset_dicts))
    logger.info(f"Selecting {num_to_visualize} random samples for visualization...")
    random_samples = random.sample(dataset_dicts, num_to_visualize)
    
    for i, image_dict in enumerate(random_samples):
        file_name = os.path.basename(image_dict["file_name"])
        logger.info(f"Processing sample {i+1}/{num_to_visualize}: {file_name}")
        
        img = cv2.imread(image_dict["file_name"])
        if img is None: continue

        inputs_for_model = {"image": torch.as_tensor(img.astype("float32").transpose(2, 0, 1)), "height": image_dict["height"], "width": image_dict["width"]}

        with torch.no_grad():
            prediction = predictor.model([inputs_for_model])[0]
            
        comparison_image = create_comparison_visualization(image_dict, prediction, metadata, cfg, coco_api)
        if comparison_image is not None:
            output_path = os.path.join(output_dir, f"comparison_{file_name}")
            cv2.imwrite(output_path, comparison_image)

    logger.info(f"Visualizations successfully saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize model predictions against ground truth for the EHOI project.")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of random samples to visualize.")
    parser.add_argument("--input-json", type=str, default=None, help="Path to the COCO JSON annotation file.")
    parser.add_argument("--output-dir", default="./visualization_output", help="Directory to save the visualization images.")
    parser.add_argument("--conf-threshold", type=float, default=0.1, help="Confidence threshold for displaying predictions.")
    
    args = parser.parse_args()
    main(args)