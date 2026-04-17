# Digitizing Nepal’s Written Heritage: A Comprehensive HTR Pipeline for Old Nepali Manuscripts

<p align="center">
</p>
<p align="center">
  <img src="app/web_app.jpeg" width="650"/>
</p>

<p align="center">
  Example of the end-to-end HTR pipeline output.<br/>
  <em>Left:</em> Sample of an Old Nepali manuscript line.
  <em>Right:</em> Predicted line-level transcription
</p>


## Environment
We use a conda Python 3.12 environment. To create and activate the env run:
```
conda create -n nepocr_env python=3.12
conda activate nepocr_env
pip install -r requirements.txt
```

## Dataset
For the training pipeline, we have a total of three stages. They are:
| Training Stage | Dataset Description | Source / Link |
|---------------|---------------------|---------------|
| Stage 1 | Synthetic Devanagari line images | link-anonymized
| Stage 2 | Printed Devanagari line images | link-anonymized
| Stage 3 | Handwritten Old Nepali manuscript images | Restricted / Not Publicly Available |



## Code Setup
```
├── README.md              <- Project overview and usage instructions
├── requirements.txt       <- Python dependencies
├── app/                   <- Code setup for interactive web app
│
├── corpus/                <- Line corpus for the datasets used, and is further used to train the tokenizer 
│
├── data/                  <- Dataset files (not pushed to git)
│
├── decoding/              <- Decoding methods implementations for the final model
│
├── fonts/                 <- Set of fonts used to generate the Old Nepali Synthetic Data (stage 1 data)
│
├── results/               <- Evaluation results of final trained models
│
├── src/                   <- Source code for training, inference, preprocessing, and evaluation
│
└── tokenizer/             <- Tokenizer training scripts and saved tokenizers
```


## Training and Evaluation Pipeline
The training pipeline follows a three-step training for all three stages mentioned above:

### Step 1: Stage 1 training
Train the model on the 105,000 Old Nepali synthetic line images to learn general script-level visual and linguistic patterns (for 6 epochs)
```
python src/trainer.py \
  --dataset_name oldNepaliSynthetic_105k_vnoisy \
  --tokenizer_type byteBPE \
  --vocab_size 500 \
  --decoder bert \
  --encoder trocr-base-handwritten
```
The model_name for this stage will be saved in our example as: *trocr-base-handwritten-BERT-oldNepaliSynthetic_105k_vnoisy-byteBPE-500*

### Step 2: Stage 2 training
The model saved from stage 1 is then used to initialize the second stage training with the  *--finetune_from_model* argument. This stage helps bridge the gap between synthetically generated line images and real data (10 epochs)
```
python src/trainer.py \
  --dataset_name nagari \
  --tokenizer_type byteBPE \
  --vocab_size 500 \
  --decoder bert \
  --encoder trocr-base-handwritten \
  --finetune_from_model models/model_name_saved_from_stage_1
```
The model_name for this stage will be saved in our example as: *trocr-base-handwritten-BERT-oldNepaliSynthetic_105k_vnoisy-byteBPE-500_finetuned_on_nagari*


### Step 3: Stage 3 training
Finally, the model obtained after stage 2 is further fine-tuned on the main handwritten Old Nepali manuscript dataset, with 8x augmentation (for 20 epochs)
```
python src/trainer.py \
  --dataset_name oldNepali_fullset_aug8 \
  --tokenizer_type byteBPE \
  --vocab_size 500 \
  --decoder bert \
  --encoder trocr-base-handwritten \
  --finetune_from_model models/model_name_saved_from_stage_2
```
The model_name for this stage will be saved in our example as: *trocr-base-handwritten-BERT-oldNepaliSynthetic_105k_vnoisy-byteBPE-500_finetuned_on_nagari*

### Step 4: Model evaluation
Run the evaluation script on a trained model by specifying the model name. Evaluation results are saved to the results/ directory
```
python src/eval.py
```


## Contact
Anonymized
