from transformers import VisionEncoderDecoderModel

model = VisionEncoderDecoderModel.from_pretrained("models/model_name")
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params / 1e6:.2f}M")