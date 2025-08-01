import json
import os
import shutil
import argparse
from tqdm import tqdm

def check_and_clean(json_path, image_root, overwrite=False):
    """
    Legge un file JSON COCO, verifica l'esistenza di ogni file immagine.
    Se overwrite=True, crea un backup e sovrascrive l'originale.
    Altrimenti, stampa solo un report dei file mancanti.
    """
    if not os.path.exists(json_path):
        print(f"ERRORE: Il file JSON non esiste, lo salto: {json_path}")
        return

    print(f"\n{'=' * 20}\nANALISI FILE: {json_path}\n{'=' * 20}")
    print(f"Usando la cartella immagini: {image_root}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)

    images_info = data.get("images", [])
    if not images_info:
        print("Nessuna immagine trovata nel file JSON. Nessuna azione richiesta.")
        return

    missing_files_info = []
    
    for image_entry in tqdm(images_info, desc=f"Verifica immagini in {os.path.basename(json_path)}"):
        file_name = image_entry["file_name"]
        
        # --- LOGICA CORRETTA E DEFINITIVA ---
        # Unisce la cartella radice delle immagini con il nome del file letto dal JSON.
        full_path = os.path.join(image_root, file_name)

        if not os.path.exists(full_path):
            missing_files_info.append(image_entry)
            
    if not missing_files_info:
        print("\n✅ Fantastico! Tutti i file immagine referenziati nel JSON esistono. Il dataset è integro.")
    else:
        print(f"\n❌ ATTENZIONE: Trovati {len(missing_files_info)} file immagine MANCANTI!")
        print("I seguenti riferimenti a immagini verranno rimossi se si procede con la pulizia:")
        for missing in missing_files_info:
            print(f"  - ID: {missing['id']}, File: {missing['file_name']}")
            
        if overwrite:
            print("\n--- AVVIO PULIZIA AUTOMATICA ---")
            backup_path = json_path + ".bak"
            try:
                shutil.copyfile(json_path, backup_path)
                print(f"Backup del file originale creato in: {backup_path}")
            except Exception as e:
                print(f"ERRORE CRITICO: Impossibile creare il backup. Operazione annullata. Dettagli: {e}")
                return

            clean_data = clean_coco_json(data, missing_files_info)
            
            try:
                with open(json_path, 'w') as f:
                    json.dump(clean_data, f, indent=4) 
                print(f"✅ Il file originale '{json_path}' è stato pulito e sovrascritto con successo.")
            except Exception as e:
                print(f"ERRORE CRITICO: Impossibile sovrascrivere il file. Ripristina dal backup '{backup_path}'. Dettagli: {e}")
        else:
            print("\nModalità di sola lettura. Nessuna modifica apportata.")
            print("Per pulire il file, riesegui lo script con il flag --overwrite.")

def clean_coco_json(data, missing_files_info):
    """
    Rimuove le immagini mancanti e le loro annotazioni da un dizionario COCO.
    """
    missing_image_ids = {img['id'] for img in missing_files_info}
    
    original_image_count = len(data['images'])
    original_annot_count = len(data['annotations'])
    
    data['images'] = [img for img in data['images'] if img['id'] not in missing_image_ids]
    data['annotations'] = [ann for ann in data['annotations'] if ann['image_id'] not in missing_image_ids]
    
    print(f"Pulizia eseguita: rimosse {original_image_count - len(data['images'])} immagini e {original_annot_count - len(data['annotations'])} annotazioni.")
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verifica e pulisce i file di annotazione COCO.")
    parser.add_argument(
        'json_files', 
        nargs='+', 
        help="Uno o più percorsi ai file JSON da analizzare (es. data/egoism-hoi-dataset/annotations/train_coco.json)"
    )
    parser.add_argument(
        '--image-root', 
        default='data/egoism-hoi-dataset/images', 
        help="Percorso alla cartella che contiene fisicamente le immagini."
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help="ATTENZIONE: Sovrascrive i file JSON originali con la versione pulita. Crea prima un backup."
    )
    args = parser.parse_args()

    for json_file in args.json_files:
        check_and_clean(json_file, args.image_root, args.overwrite)