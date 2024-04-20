import torch
from PIL import Image
import open_clip
from gradio import Interface, inputs, outputs

def start():
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    return model, preprocess, tokenizer

def process(model, preprocess, tokenizer, image_path, text):
    if isinstance(image_path, str):
        image = Image.open(image_path)
        image = preprocess(image.resize((512, 512))).unsqueeze(0)
    else:
        image = preprocess(image_path.resize((512, 512))).unsqueeze(0)

    text = tokenizer([text])
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (image_features @ text_features.T) * 100
    return similarity

def predict(image, text):
    model, preprocess, tokenizer = start()
    similarity = process(model, preprocess, tokenizer, image, text)
    return similarity.item()

inputs = [
    inputs.Image(type="pil", label="Image"),
    inputs.Textbox(label="Text")
]

outputs = outputs.Textbox(label="Similarity")

# If you want to run it locally, you can use the following code :(
Interface(fn=predict, inputs=inputs, outputs=outputs).launch()
# If you want to share it online, you can use the following code :)
# Interface(fn=predict, inputs=inputs, outputs=outputs).launch(share=True)