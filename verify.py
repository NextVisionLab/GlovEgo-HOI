# visualize_empty_seg.py
import json
import cv2
import numpy as np
import os

def visualize_empty_segmentation(json_path, image_root):
    print(f"--- Finding annotations with empty segmentations in: {json_path} ---")
    
    with open(json_path, 'r') as f:
        data = json.load(f)

    annotations = data.get('annotations', [])
    images_info = {img['id']: img for img in data.get('images', [])}
    categories_info = {cat['id']: cat['name'] for cat in data.get('categories', [])}

    found_count = 0
    
    for ann in annotations:
        if 'segmentation' in ann and not ann['segmentation']:
            image_id = ann['image_id']
            if image_id in images_info:
                image_data = images_info[image_id]
                image_path = os.path.join(image_root, image_data['file_name'])
                
                if not os.path.exists(image_path):
                    print(f"[WARNING] Image not found at: {image_path}")
                    continue
                
                print(f"\nFound annotation with empty segmentation:")
                print(f"  - Image: {image_path}")
                print(f"  - Annotation ID: {ann['id']}")
                print(f"  - Category: {categories_info.get(ann['category_id'], 'Unknown')}")
                print(f"  - BBox: {ann['bbox']}")

                image = cv2.imread(image_path)
                x, y, w, h = [int(v) for v in ann['bbox']]
                
                # Disegna il bounding box
                cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)
                
                # Aggiungi etichetta
                label = f"ID: {ann['id']}, Cat: {categories_info.get(ann['category_id'])}"
                cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)
                
                cv2.imshow('Image with Empty Segmentation', image)
                print("Press any key to continue to the next one...")
                cv2.waitKey(0)
                
                found_count += 1

    cv2.destroyAllWindows()
    print(f"\n--- Visualization Complete. Found {found_count} instances. ---")

if __name__ == "__main__":
    json_file = 'data/egoism-hoi-dataset/annotations/train_coco.json'
    # ATTENZIONE: Assicurati che questo percorso alla cartella delle immagini sia corretto
    image_folder = 'data/egoism-hoi-dataset/images/'
    visualize_empty_segmentation(json_file, image_folder)