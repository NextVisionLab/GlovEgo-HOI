import torch
import random
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.data.ehoi_dataset_mapper_v1 import EhoiDatasetMapperDepthv1  # Assicurati che il nome e l'import siano corretti

def debug_mapper():
    # 1. Registra il tuo dataset (usa il file di training)
    train_json = "data/egoism-hoi-dataset/annotations/train_coco.json"  # <-- METTI QUI IL PERCORSO CORRETTO
    images_path = "data/egoism-hoi-dataset/images/" # <-- METTI QUI IL PERCORSO CORRETTO
    
    register_coco_instances("dataset_debug", {}, train_json, images_path)
    dataset_dicts = DatasetCatalog.get("dataset_debug")
    metadata = MetadataCatalog.get("dataset_debug")
    
    # Imposta i metadati dei keypoint, sono necessari
    KEYPOINT_NAMES = [
        "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
        "index_mcp", "index_pip", "index_dip", "index_tip",
        "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
        "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
        "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"
    ]
    metadata.set(keypoint_names=KEYPOINT_NAMES, keypoint_flip_map=[])

    # 2. Carica la configurazione
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file("./configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
    cfg.merge_from_file("./configs/Custom/custom.yaml") # Il tuo file custom
    # Aggiungi eventuali altre configurazioni necessarie al mapper
    # ...

    # 3. Istanzia il mapper
    # Potrebbe richiedere 'data_anns_sup', passaglielo se necessario
    import json
    with open(train_json) as f:
        data_anns_train_sup = json.load(f)
        
    mapper = EhoiDatasetMapperDepthv1(cfg, data_anns_sup=data_anns_train_sup, is_train=True)

    # 4. Prendi un campione a caso ed esegui il mapper
    random_sample = random.choice(dataset_dicts)
    print("--- Input al Mapper (estratto da dataset_dicts) ---")
    print(random_sample)
    
    data = mapper(random_sample)

    # 5. Ispeziona l'output
    print("\n--- Output del Mapper ---")
    print(data)
    
    # Il controllo cruciale
    print("\n--- Ispezione Campo 'instances' ---")
    if "instances" in data:
        instances = data["instances"]
        print(f"Tipo di 'instances': {type(instances)}")
        
        # Stampa tutti i campi presenti nell'oggetto Instances
        print(f"Campi presenti: {instances.get_fields().keys()}")
        
        if instances.has("gt_keypoints"):
            print("\n!!! SUCCESSO: Il campo 'gt_keypoints' è PRESENTE. !!!")
            gt_keypoints = instances.gt_keypoints
            print(f"Tipo di 'gt_keypoints': {type(gt_keypoints)}")
            print(f"Shape di gt_keypoints.tensor: {gt_keypoints.tensor.shape}")
            print("Esempio di valori:")
            print(gt_keypoints.tensor)
        else:
            print("\n!!! ERRORE: Il campo 'gt_keypoints' è ASSENTE. !!!")
            
    else:
        print("!!! ERRORE: Nessun campo 'instances' nell'output del mapper. !!!")


if __name__ == "__main__":
    debug_mapper()