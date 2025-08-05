import json

with open("data/oldNepali_fullset_binarized/labels/labels_train.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for entry in data:
    path = entry["image_path"]

    if "data/oldNepali_fullset" in path:
        path = path.replace("data/oldNepali_fullset", "data/oldNepali_fullset_binarized", 1)

    # elif "datasetProcessedAugmented/" in path:
    #     path = path.replace("datasetProcessedAugmented/", "datasetProcessedAugmented/train/", 1)

    entry["image_path"] = path

with open("data/oldNepali_fullset_binarized/labels/labels_train.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("done")
