import torch
from transformers import TrainerCallback

class PrintPredictionsCallback(TrainerCallback):
    def __init__(self, sample_batch, tokenizer, print_every=100):
        self.samples = sample_batch
        self.tokenizer = tokenizer
        self.print_every = print_every

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.print_every == 0 and state.global_step > 0:
            print(f"\n--- Step {state.global_step} ---")
            model = kwargs["model"]
            model.eval()
            device = model.device

            pixel_values = torch.stack([s["pixel_values"] for s in self.samples]).to(device)
            labels = torch.stack([s["labels"] for s in self.samples]).to(device)

            with torch.no_grad():
                predictions = model.generate(pixel_values)

            #  predictions
            decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)

            # ground truth labels (we convert -100 back to pad_token)
            labels = labels.clone()
            labels[labels == -100] = self.tokenizer.pad_token_id
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

            for i, (pred, true) in enumerate(zip(decoded_preds, decoded_labels)):
                print(f"Sample {i+1}")
                print(f"Prediction   : {pred}")
                print(f"Ground Truth : {true}")
                print("")

            model.train()

        return control