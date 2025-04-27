import json

with open("data/oldNepali/original/test/labels.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for entry in data:
    path = entry["image_path"]

    if "data/oldNepali/clean/" in path:
        path = path.replace("data/oldNepali/clean/", "data/oldNepali/original/", 1)

    # elif "datasetProcessedAugmented/" in path:
    #     path = path.replace("datasetProcessedAugmented/", "datasetProcessedAugmented/train/", 1)

    entry["image_path"] = path

with open("data/oldNepali/original/test/labels_processed.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)