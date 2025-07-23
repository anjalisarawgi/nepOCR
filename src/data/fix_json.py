import json

with open("data/oldNepali_fullset/labels_fixed.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for entry in data:
    path = entry["image_path"]

    if "oldNepali_100/processed/images" in path:
        path = path.replace("oldNepali_100/processed/images", "data/oldNepali_fullset/oldNepali_100/images", 1)

    # elif "datasetProcessedAugmented/" in path:
    #     path = path.replace("datasetProcessedAugmented/", "datasetProcessedAugmented/train/", 1)

    entry["image_path"] = path

with open("data/oldNepali_fullset/labels_fixed.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)



# ############################################
# import json

# with open('data/oldNepaliDataset3/labels_train.json', 'r', encoding='utf-8') as f:
#     data = json.load(f)

# for item in data:
#     if "label" in item:
#         item["text"] = item.pop("label")

# with open('data/oldNepaliDataset3/labels_train_processed.json', 'w', encoding='utf-8') as f:
#     json.dump(data, f, ensure_ascii=False, indent=2)

# print("✅ Labels changed to text successfully!")