import requests
from PIL import Image
import torch

MODELS = ["clip", "llava", "qwen"]

def _load_image(url: str) -> Image.Image:
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    return Image.open(resp.raw).convert("RGB")


def process(model_name: str, image_url: str, labels: list[str]) -> dict:
    """Entry function.

    Args:
        model_name: one of 'clip', 'llava', 'qwen'
        image_url: URL to an image
        labels: list of label strings, e.g. ['cat','dog']

    Returns:
        Dict mapping label -> probability (float)
    """
    model_name = model_name.lower()

    if not model_name in MODELS:
        raise TypeError(f"model_name value '{model_name}' is not in {MODELS}")

    if model_name == "clip":
        return _process_clip(image_url, labels)
    if model_name == "llava":
        return _process_llava(image_url, labels)
    if model_name == "qwen":
        return _process_qwen(image_url, labels)
    raise ValueError(f"Unknown model_name: {model_name}")


def _process_clip(image_url: str, labels: list[str]) -> dict:
    from transformers import CLIPProcessor, CLIPModel

    image = _load_image(image_url)
    model_id = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)

    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    # logits_per_image shape: (batch, num_text)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)[0]
    return {label: float(prob) for label, prob in zip(labels, probs.tolist())}


def _process_llava(image_url: str, labels: list[str]) -> dict:
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    model_id = "llava-hf/llava-1.5-7b-hf"
    if device == "cuda":
        model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype, device_map="auto")
    else:
        model = LlavaForConditionalGeneration.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)

    image = _load_image(image_url)
    prompt = f"USER: <image>\nWhich of the following is depicted: {', '.join(labels)}? Answer with one word. ASSISTANT:"

    inputs = processor(text=prompt, images=image, return_tensors="pt")
    if device == "cuda":
        inputs = inputs.to(device, dtype)
    else:
        inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        last_token_logits = outputs.logits[:, -1, :]

    tokenizer = processor.tokenizer
    token_ids = []
    for lab in labels:
        enc = tokenizer.encode(lab, add_special_tokens=False)
        if len(enc) == 0:
            raise ValueError(f"Label '{lab}' tokenized to empty sequence")
        token_ids.append(enc[0])

    label_logits = last_token_logits[0, token_ids]
    probs = torch.softmax(label_logits.float(), dim=0)
    return {label: float(p) for label, p in zip(labels, probs.tolist())}


def _process_qwen(image_url: str, labels: list[str]) -> dict:
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    model_id = "Qwen/Qwen2-VL-7B-Instruct"
    if device == "cuda":
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype, device_map="auto")
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)

    image = _load_image(image_url)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": f"Which of the following is depicted: {', '.join(labels)}? Answer with one word."},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    if device == "cuda":
        inputs = inputs.to(device)
    else:
        inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        last_token_logits = outputs.logits[:, -1, :]

    tokenizer = processor.tokenizer
    token_ids = []
    for lab in labels:
        enc = tokenizer.encode(lab, add_special_tokens=False)
        if len(enc) == 0:
            raise ValueError(f"Label '{lab}' tokenized to empty sequence")
        token_ids.append(enc[0])

    label_logits = last_token_logits[0, token_ids]
    probs = torch.softmax(label_logits.float(), dim=0)
    return {label: float(p) for label, p in zip(labels, probs.tolist())}

# Simple example
example_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
labels = ["cat", "dog"]
try:
    res = process("clip", example_url, labels)
    print("CLIP:", res)
except Exception as e:
    print("CLIP run failed:", e)

res = process("llava", example_url, labels)
print("LLAVA:", res)

res = process("qwen", example_url, labels)
print("QWEN:", res)