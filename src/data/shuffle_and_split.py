import json
import random
import argparse
import os

def load_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_labels(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def split_data(data, seed, train_frac=0.8, val_frac=0.1):
    random.seed(seed)
    random.shuffle(data)

    n = len(data)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)
    n_test  = n - n_train - n_val  # whatever remains

    train_data = data[:n_train]
    val_data   = data[n_train : n_train + n_val]
    test_data  = data[n_train + n_val :]

    print(f"Total: {n} → Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    return train_data, val_data, test_data

def main():
    input_dir = "data/oldNepali_fullset/labels_normalized.json"
    output_dir = "data/oldNepali_fullset/labels"
    data = load_labels(input_dir)
    train, test, val = split_data(data, seed=41)

    save_labels(train, os.path.join(output_dir, "labels_train.json"))
    save_labels(val,   os.path.join(output_dir, "labels_val.json"))
    save_labels(test,  os.path.join(output_dir, "labels_test.json"))

if __name__ == "__main__":
    main()