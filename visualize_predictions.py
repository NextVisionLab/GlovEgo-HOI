import argparse
import os
import random
import cv2
import torch
import numpy as np
from typing import List, Dict

# Import utilities from Detectron2 and a_ehoi_core
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.utils.visualizer import Visualizer, ColorMode, VisImage
from detectron2.data.datasets import register_coco_instances
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

# Assicurati che i moduli custom siano importabili
# Questo presuppone che lo script sia eseguito dalla root del progetto
from detectron2.modeling.meta_arch import MMEhoiNetv1

# Helper function to draw keypoints with connections
def draw_keypoints(image, keypoints_tensor, keypoint_names, skeleton, color=(0, 255, 0)):
    """
    Disegna i keypoint e lo scheletro su un'immagine.
    keypoints_tensor: un tensore [K, 3] con (x, y, vis/score) per K keypoint.
    """
    # Definiamo le connessioni dello scheletro della mano
    # (Questo è un esempio, puoi adattarlo al tuo scheletro)
    HAND_SKELETON = [
        [0, 1], [1, 2], [2, 3], [3, 4],      # Pollice
        [0, 5], [5, 6], [6, 7], [7, 8],      # Indice
        [0, 9], [9, 10], [10, 11], [11, 12], # Medio
        [0, 13], [13, 14], [14, 15], [15, 16],# Anulare
        [0, 17], [17, 18], [18, 19], [19, 20] # Mignolo
    ]

    keypoints = keypoints_tensor.cpu().numpy()
    
    # Disegna le connessioni
    for p1_idx, p2_idx in HAND_SKELETON:
        # Assicurati che gli indici siano validi
        if p1_idx < len(keypoints) and p2_idx < len(keypoints):
            p1 = keypoints[p1_idx]
            p2 = keypoints[p2_idx]
            # Disegna la linea solo se entrambi i keypoint sono visibili/predetti con confidenza
            if p1[2] > 0.1 and p2[2] > 0.1:
                cv2.line(image, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, 2)

    # Disegna i keypoint
    for i, (x, y, v) in enumerate(keypoints):
        if v > 0.1:
            cv2.circle(image, (int(x), int(y)), 4, color, -1)


def visualize_comparison(image_dict, prediction, metadata):
    """
    Crea un'immagine che confronta Ground Truth e Predizioni.
    """
    img = cv2.imread(image_dict["file_name"])
    h, w, _ = img.shape
    
    # Crea due copie dell'immagine per GT e Predizione
    gt_img = img.copy()
    pred_img = img.copy()

    # --- Visualizzazione Ground Truth ---
    gt_visualizer = Visualizer(gt_img, metadata, scale=1.0)
    gt_vis_output = gt_visualizer.draw_dataset_dict(image_dict)
    gt_img_processed = gt_vis_output.get_image()
    # Aggiungi titolo
    cv2.putText(gt_img_processed, "Ground Truth", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    # --- Visualizzazione Predizioni ---
    pred_visualizer = Visualizer(pred_img, metadata, scale=1.0, instance_mode=ColorMode.IMAGE)
    instances = prediction["instances"].to("cpu")
    pred_vis_output = pred_visualizer.draw_instance_predictions(instances)
    pred_img_processed = pred_vis_output.get_image()

    # Aggiungi le nostre predizioni custom (lato, stato, keypoint)
    hand_instances = instances[instances.pred_classes == metadata.hand_id]
    additional_outputs = prediction.get("additional_outputs")
    if additional_outputs:
        hand_preds = additional_outputs
        
        for i in range(len(hand_preds)):
            box = hand_preds.boxes[i].numpy().astype(int)
            side = "Right" if hand_preds.sides[i].item() == 1 else "Left"
            contact = "Contact" if hand_preds.contact_states[i].item() == 1 else "No-Contact"
            score = hand_preds.scores[i].item()
            
            label = f"Hand: {side}, {contact} ({score:.2f})"
            cv2.putText(pred_img_processed, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Disegna i keypoint predetti, se esistono
    if hand_instances.has("pred_keypoints"):
        for kps in hand_instances.pred_keypoints:
             draw_keypoints(pred_img_processed, kps, metadata.keypoint_names, metadata.keypoint_connection_rules, color=(0, 255, 0))

    # Aggiungi titolo
    cv2.putText(pred_img_processed, "Prediction", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    # Unisci le due immagini fianco a fianco
    comparison_img = np.concatenate((gt_img_processed, pred_img_processed), axis=1)
    
    return comparison_img


def main(args):
    # --- 1. Caricamento Configurazione e Metadati ---
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(args.config_file)
    cfg.MODEL.WEIGHTS = args.weights_file
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.conf_threshold # Imposta una soglia per visualizzare solo predizioni confidenti
    cfg.freeze()

    # --- 2. Registrazione e Caricamento Dataset ---
    dataset_name = "ehoi_viz_dataset"
    json_path = args.json_file
    images_path = os.path.dirname(json_path)

    register_coco_instances(dataset_name, {}, json_path, images_path)
    
    metadata = MetadataCatalog.get(dataset_name)
    # Aggiungi informazioni custom che potrebbero mancare se non si usa il nostro train script
    if not hasattr(metadata, 'thing_classes'):
        # Ricavale dal file json
        import json
        with open(json_path, 'r') as f:
            data = json.load(f)
        metadata.thing_classes = [cat['name'] for cat in data['categories']]
    
    hand_id = -1
    for i, name in enumerate(metadata.thing_classes):
        if name in ["hand", "mano"]:
            hand_id = i
            break
    metadata.hand_id = hand_id

    # Assicurati che i metadati dei keypoint siano presenti
    if not hasattr(metadata, 'keypoint_names'):
         KEYPOINT_NAMES = [
            "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
            "index_mcp", "index_pip", "index_dip", "index_tip",
            "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
            "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
            "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"
        ]
         metadata.keypoint_names = KEYPOINT_NAMES
         metadata.keypoint_connection_rules = [] # La nostra funzione `draw_keypoints` ha la sua logica
    
    dataset_dicts = DatasetCatalog.get(dataset_name)

    # --- 3. Creazione Modello e Caricamento Pesi ---
    model = build_model(cfg)
    model.eval()
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    # --- 4. Selezione Immagini e Inferenza ---
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Seleziona N immagini casuali
    random_samples = random.sample(dataset_dicts, min(args.num_samples, len(dataset_dicts)))
    
    print(f"Generating {len(random_samples)} visualizations...")
    for i, image_dict in enumerate(random_samples):
        print(f"  Processing image {i+1}/{len(random_samples)}: {os.path.basename(image_dict['file_name'])}")
        
        # Carica immagine
        img = cv2.imread(image_dict["file_name"])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Esegui inferenza
        with torch.no_grad():
            # Il modello si aspetta un batch di input
            inputs = [{"image": torch.as_tensor(img_rgb.transpose(2, 0, 1)), "height": img.shape[0], "width": img.shape[1]}]
            predictions = model(inputs)
            # Prendiamo la prima (e unica) predizione dal batch
            prediction = predictions[0]

        # Crea e salva l'immagine di confronto
        comparison_image = visualize_comparison(image_dict, prediction, metadata)
        output_path = os.path.join(output_dir, f"comparison_{os.path.basename(image_dict['file_name'])}")
        cv2.imwrite(output_path, comparison_image)

    print(f"\nVisualizations saved in: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize model predictions against ground truth.")
    parser.add_argument("--config-file", required=True, help="Path to the model's .yaml config file.")
    parser.add_argument("--weights-file", required=True, help="Path to the model's .pth weights file.")
    parser.add_argument("--json-file", required=True, help="Path to the COCO JSON file of the dataset to visualize.")
    parser.add_argument("--output-dir", default="./visualization_output", help="Directory to save the output images.")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of random samples to visualize.")
    parser.add_argument("--conf-threshold", type=float, default=0.5, help="Confidence threshold for predictions.")
    
    args = parser.parse_args()
    main(args)