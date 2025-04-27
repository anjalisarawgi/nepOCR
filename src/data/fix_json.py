import json

with open("data/oldNepaliSynthetic30k/labels.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for entry in data:
    path = entry["image_path"]

    if "oldNepaliSynthetic_30k/" in path:
        path = path.replace("oldNepaliSynthetic_30k/", "data/oldNepaliSynthetic30k/", 1)

    # elif "datasetProcessedAugmented/" in path:
    #     path = path.replace("datasetProcessedAugmented/", "datasetProcessedAugmented/train/", 1)

    entry["image_path"] = path

with open("data/oldNepaliSynthetic30k/labels_processed.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)