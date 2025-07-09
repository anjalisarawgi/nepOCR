# nepOCR Training Script

## Overview

This repo builds and tests a custom decoding method for OCR focused on Old Nepali scripts.  
We use a **three-stage fine-tuning pipeline**:

- **Stage 1: Pretraining the Decoder**  
  Synthetic data is generated using text corpora and fonts from the `fonts/` directory, and the PIL library.  
  The dataset is stored in the folder `data/oldNepaliSynthetic*`, with sizes the ranging from **10k to 200k samples**. (we can experiment with these as we like)

- **Stage 2: Fine-tuning on Nagari Scripts**  
  Fine-tuning on the **Nagari** dataset (`data/nagari`), provided by the University of Heidelberg.  
  Dataset size: approximately **5,000 samples**.

- **Stage 3: Fine-tuning on Old Nepali Scripts**  
  Final fine-tuning on the **Old Nepali** dataset (`data/oldNepali`), also provided by the University of Heidelberg.  
  This is the **main dataset**, with around **1,500 samples**.

---


## Dataset setup
Please download it from this link: [Link](https://drive.google.com/drive/folders/1uDNqMMBWFBVD7Vstwm_WrVJlB5GjAxz0?usp=sharing)

---
## 🚀 How to Run

Training is done through one script which is:
> `src/trainer.py`

### Stage 1: Pretraining the Decoder

Run:

```bash
python src/trainer.py --dataset_name oldNepaliSynthetic10k
```  

- please change the dataset_name as needed (e.g., oldNepaliSynthetic30k, etc.).
- the trained models will be saved directly inside the models/ directory.
- we can also customize the encoder, decoder, tokenizer type, and vocabulary size using additional arguments as required


###  Stage 2 and Stage 3: Fine-tuning on Real Datasets
To fine-tune the pretrained model:
```bash
python src/trainer.py --dataset_name nagari --finetune_from_model models/<pretrained_model_name>
```
- Replace <pretrained_model_name> with the directory name of your pretrained model.
- For Stage 3, set --dataset_name oldNepali.
- Fine-tuned models are saved inside the models/ directory.  




## 🚀 How to Run Streamlit app
Run:

```bash
streamlit run app.py
```  

- please run the following to upload the model on hugging face (???)
> `src/trainer.py`
