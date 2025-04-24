import torch 
from transformers import TrainerCallback 

class PrintPredictionsCallback(TrainerCallback):
    def __init__(self, sample_batch, tokenizer, print_every = 100):
        self.samples = sample_batch
        self.tokenizer = tokenizer
        self.print_every = print_every

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.print_every == 0 and state.global_step > 0:
            print(f"\n--- Step {state.global_step} ---")
            pixel_values = torch.stack([s["pixel_values"] for s in self.samples]).to(kwargs["model"].device)
            with torch.no_grad():
                predictions = kwargs["model"].generate(pixel_values)
            decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens = True)
            for i, pred in enumerate(decoded_preds):
                print(f"Sample {i+1} prediction: {pred}")
                
        return control 