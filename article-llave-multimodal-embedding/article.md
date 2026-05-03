# From CLIP to LLaVE: Smarter Training for Multimodal Embeddings

## Introduction

Large Multimodal Models (LMMs) have become an industry-standard for daily AI users. For example, Claude and Gemini models are able to work with text, image, voice etc. simultaneously for a larger scope of usability. 

Multimodal models are based on a fundamental mechanism called **embedding**. Embedding is a the process of converting data into vectors, where each dimension of the vector encodes some characteristic of the data. These embeddings share the vector space between different data modalities. For example, a vector for the text "scary clown" will be very similar to the image of Pennywise. 

This multimodal approach allows for various applications, such as text-to-image or image-to-text search, document analysis, medical diagnostics and robotics control. However, the performance of LMMs is significantly based on the quality of the embedding. 

**LLaVE** (Large Language and Vision Embedding Model) became an innovation in training multimodal embeddings introduced by Tencent. In this article, we will understand how those embeddings are learned, what was lacking in previous approaches, and how LLaVE brought a state-of-the-art performance to multimodal embedding. 

## CLIP: First Vision-Language Model

**CLIP**, short for **Contrastive Language-Image Pre-training**, was released by OpenAI team in 2021 and since then has become a pillar in multimodal AI development. Its true innovation stemmed from the fact that it consists of two *separate* transformers (one for text embedding and another for image embedding) that share *the same* embedding space. 

The fact that CLIP works with language and images explains the L and I letters from its name. What about the word *contrastive*? In a nutshell, **contrastive learning** means that the model learns to distinguish similar data from distant data. The embedding space of a contrastive learning model results in the fact that similar notions are geometrically close to each other, while unrelated concepts are further apart. 

Let's take a particular case of contrastive learning called **triplet loss**. Mathematically, it is defined as:

$$
L_{\text{triplet}} = \max\left(0,\; \|a - p\|_2^2 - \|a - n\|_2^2 + \text{margin}\right)
$$

In the case of triplet loss, we have a set of three examples: *anchor* $a$, *positive* $p$ and *negative* $n$. Let's see what those notions means based on example (image below).

- Anchor $a$ is an example that we want to learn from. In the image below, we want to understand that the anchor image is a dog.
- Positive $p$ is an example that we find very similar to anchor. In our example, this can be an image of a similar dog, or the anchor image with some effects applied on it.
- Negative $n$ would be something that should not be attributed to the anchor class. This can be any other image from the dataset, except for the images similar to the anchor.

Margin is a parameter that enfoces a safety buffer between positives and negatives

The goal of triplet loss is to minimize the distance between anchor and positive and to maximize the distance between anchor and negative. Here, the goal would be to minimize the distance between the black-and-white dog image (positive) and the original dog image (anchor), and minimize the distance between the owl (negative) and the original dog image (anchor).

![TODO: upload the image to different site](https://miro.medium.com/v2/resize%3Afit%3A720/format%3Awebp/1%2A7xMtz23e_U1xfqVWYrsevA.png)

CLIP took it one step further and implemented **multi-class N-pair loss**, which means that instead of having one positive and one negative for each anchor, one anchor is supplied with N positives and N negatives. In this approach, the goal becomes to take all possible pairs of negatives and positives (except for when negative and positive image coincide) and perform contrastive learning on those pairs. This allows for batch training on many pairs simultaneously and provides better guidance for loss function.

Now, though CLIP's approach sounds totally valid, it still has some practical limitations. First limitation stems from the fact that all negative examples are treated equally. Hard examples should have a higher influence over the training process than easy examples. Since both easy and hard examples have the same learning magnitude, the model does not learn fine-grained differences between the two. Second limitation is that due to CLIP's multi-class N-pair loss, the learning process still comes from from one positive and one negative per anchor. One negative might not provide enough signal to learn granular distinction. Having more negative samples simultaneously would improve the direction in which the anchor embedding should be placed, since the resulting embedding would be now further not only from one negative sample, but from many others too.

## InfoNCE Loss: Improvement over Triplet Loss
**InfoNCE** (Information Noise Contrastive Estimation) is an improved loss function that compares anchor to one positive and many negatives *at the same time*. 

$$
\mathcal{L}_{\text{InfoNCE}} = - \log \frac{\exp\left(\mathrm{sim}(a, p) / \tau \right)}{\exp\left(\mathrm{sim}(a, p) / \tau \right) + \sum_{i=1}^{N} \exp\left(\mathrm{sim}(a, n_i) / \tau \right)}
$$
where $sim(x, y)$ determines how similar $x$ and $y$ are, and $\tau$ is the temperature parameter.

Temperature $\tau$ regulates how much the model focuses on hard examples. Small values of $\tau$ will result in a larger difference between smaller values, resulting in the model focusing more on hard samples. Conversely, large values of $\tau$ will result in a more uniform learning across the samples. Note that $\tau$ does not fully resolve the first limitation of CLIP. Even though we can now more clearly accentuate the difference between similar samples, there is still no way to define which samples are hard and which ones are easy. 

InfoNCE can combine the differences over many negatives to provide a better direction for the anchor. Due to this property, InfoNCE loss is used extensively in contrastive learning, natural language processing (NLP) and recommendation systems. However, one issue persists: the negatives are treated uniformly, which becomes problematic if the goal is to train a model that can differentiate between similar samples. 

## LLaVE for Multimodal Embedding
Now, having a brief overview of previous approaches, we can have a greater understanding of the innovation behind LLaVE. In March 2026, LLaVE-style models brought **weighted contrastive loss** to multimodal models. They address the issue of uniform weight across negative samples by assigning larger weight to harder samples, thus improving learning signal, reducing noise and dataset bias. 

Intuitively, the question becomes: how does the model know which samples are supposed to be learned from (hence, assign them higher weight) and which ones are just noise? For this purpose, LLaVE uses two concepts from Reinforcement Learning (RL): **policy model** $r_\pi$ and **reward model** $r_\theta$.

### Reward Model $r_\theta$
The **reward model** $r_\theta$ is a trained model that returns a reward, based on a certain input. In a general case, the formula for it looks like:

$$
r_\theta(x) \rightarrow \R
$$

where each input $x$ is mapped to some reward. In the case of embedding, the reward is given depending on the *semantic similarity* between two embedding vectors. So, $x$ would be a vector embedding pair and the reward would be how similar their meaning is.

The way that the reward model $r_\theta$ is trained is based on a concept called **reinforcement learning with human feedback**, RLHF for short. It is a supervised learning model, which learns from examples given by humans (or, in the modern age, by other AIs). The annotator is given an image and two labels (or, vice versa, a label and two images) and is asked which pair makes more sense. These pairs of preferred vs. rejected samples make up for a dataset, from which the model can learn to give a reward.

### Policy Model $r_\pi$

The **policy model** $r_\pi$ creates the embeddings for the positive and negative samples and passes those embeddings to the reward model $r_\theta$. Remember that the reward means how similar one embedding is to another. However, we have only one positive sample and the rest of them are negatives. That means that the negative samples that carry on similar meaning to the positive samples are **hard negatives**, so the model should learn to distinguish those two embeddings better. **The reward produced for negative samples directly becomes the difficulty of those samples, hence the policy model should focus on distinguishing them better.** 

### Arriving to LLaVE Loss Function

Now, let's try to formalize the general understanding behind LLaVE's approach into a mathematical model.

Let's take the simplest example. Recall that we have our **policy model** $r_\pi$, in our case being the embedding model. Our embedding model can convert both text and images to embedding vectors. Having a image description $q_1$ and two candidate images $t_1$ (the preferred one) and $t_2$ (a negative sample), we want for our model to prefer $t_1$ over $t_2$. In this case, our loss function will look like: 

$$
\mathcal{L}_1 = -\log \frac{e^{r_\pi(q_1, t_1)}}{e^{r_\pi(q_1, t_1)} + e^{r_\pi(q_1, t_2)}}
$$

Mathematically, we see that the more the embedding model prefers $t_2$ over $t_1$, the more it will be punished, so it is incentivized to learn to associate $q_1$ with $_1$. 

Now we want to match our loss functions with InfoNCE, so that we can learn from multiple pairs at a time. Assume we have matching text-image pairs $(q_i, t_i)$, where $i=1 \ ... \ N$. We can rewrite the formula so that our policy model is rewarded for matching a given text-image pair and to punish a query being mapped closely to other images. Our formula becomes:

$$
\mathcal{L}_i = -\log \frac{e^{r_\pi(q_i, t_i)}}{e^{r_\pi(q_i, t_i)} + \sum_{j \ne i}^{N} e^{r_\pi(q_i, t_j)}}
$$

We set $j \ne i$, otherwise the model would be punished for correctly matching query $q$ to target image $t$. In this case, our model is punished for *any* similarity to the other pairs. Even if the embedding is matching closely just one text-image pair, it will be punished as if it was wrong on a smaller scale on all of them. 

Now, we introduced a loss formula for **N-pair weighted loss**. In general case, it would look like:

$$
\mathcal{L}_i = -\log \frac{e^{r_\pi(q_i, t_i)}}{e^{r_\pi(q_i, t_i)} + \sum_{j \ne i}^{N} w_{ij} \ · \ e^{r_\pi(q_i, t_j)}}
$$

where $w_{ij}$ represents the weight of learning difficulty. 

We have already established that the learning difficulty is estimated by our reward model $r_\theta$. That means we can define $w_{ij} = e^{r_\theta(q_i, t_j)}$. 

Finally, we can define the loss function used by LLaVE:

$$
\mathcal{L}_i = -\log \frac{e^{r_\pi(q_i, t_i)}}{e^{r_\pi(q_i, t_i)} + \sum_{j \ne i}^{N} e^{\left(r_\pi(q_i, t_j) + r_\theta(q_i, t_j)\right)}}
$$

Now we have made a big journey starting from triplet loss, arriving all the way up to the weighted loss function used by LLaVE!

### Combined Policy-Reward Approach

There is another clever trick that was used LLaVE development. Notice that we have both the policy model and the reward model in the loss function. If we run backpropagation, that means we have to teach two different models simultaneously. How to avoid having to train two models?

Recall that the policy model $r_\pi$ takes two embeddable inputs and produces vector embeddings for both of them, meanwhile the reward model $r_\theta$ takes two embeddings and produces a reward, equivalent to the semantic similarity between the embeddings. What LLaVE researches did is they **combined reward and policy models**, creating a model that produces both embeddings and rewards. How did they do it? 

Let's see how they reinvented the reward model $r_\theta$. Since it takes two embeddings and produces a reward score, we can make the reward model as a *wrapper* of the policy model. What that means is that the reward model operates directly on the output of the policy model to determine the similarity of the two inputs. In a general scenario, reward model becomes:

$$
r_\theta = f(r_\pi)
$$

What is left to define is the function $f$, which converts embedding vectors into similarity. Those familiar with vector databases or linear algebra can point out that similarity function between two vectors can be as simple as *cosine similarity*, which is defined as:

$$
sim(q_i, t_i) = cos \ \theta =  \frac{q_i \ · \ t_i}{||q_i|| \ · \ ||t_i||}
$$

Function $f$ can also be defined as something more complex, for example a linear regression model or a multilayer perceptron (MLP), so RLHF can still be applied to train the reward function.

Now, since the backpropagation is already applied to the policy model $r_\pi$, we do not need to apply it on the reward model $r_\theta$. 

## Comparing Multimodal Models
Now that we know what makes up a multimodal model, let's try putting a few of them to the test! For the sake of comparison, three different models were chosen: OpenAI's CLIP ViT Large, LLaVA 1.5 7b and Qwen2 VL 7B Instruct. These three models use three different approaches, so we can see how each one works.

The easiest way to get started is to use a Google Colab environment to run the test. Its virtual environment already includes the most handy ML libraries, including torch and transformers.

Let's import global libraries and define some helper variables and functions:

```py
import requests
from PIL import Image
import torch

MODELS = ["clip", "llava", "qwen"]

def _load_image(url: str) -> Image.Image:
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    return Image.open(resp.raw).convert("RGB")
```

Here we define the main entry to our models. The parameters for this function are:
- `model_name` (string) - one of the model defined in `MODELS` list
- `image_url` (string) - link to the image that we want to aggregate
- `labels` (string[]) - a list of labels that we want to compare the image against

It returns a dictionary, where the keys are labels from `labels` and values are the probabilities of the given label being attributed to the image.

```py
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
        raise ValueError(f"model_name value '{model_name}' is not in {MODELS}")

    if model_name == "clip":
        return _process_clip(image_url, labels)
    if model_name == "llava":
        return _process_llava(image_url, labels)
    if model_name == "qwen":
        return _process_qwen(image_url, labels)

    raise ValueError(f"No handler for model '{model_name}'")
```

Now let's see how each handler boots up and utilizes the model. Start off with CLIP. It is the easiest and the fastest model to run. We just pass it the loaded image and the labels, then apply softmax function to get normalized probabilities. We return the desired labels together with their probabilities.

```py
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
```

For LLaVA model, we use a special pre-defined class `LlavaForConditionalGeneration`, which initializes the LLaVA model and can load the pretrained weights. What we can see is that we define a prompt, with which we can get the response from the model. When asking it to respond with one word, what we are effectively doing is asking the model to generate the label probabilities, since it is trying to predict the next token (which is the label that we are asking it to respond with). Lastly, we retrieve the predicted tokens' probabilities and return them as a dict.

```py
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
```

For QWEN2 VL model, the approach is similar. We load the pretrained model, define the chat contents with the message and the prompt, and then retrieve the last token logits to get the probabilities of the labels.

```py
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
```

Let's try a simple example of how this code can be applied. We can try out the code in the following way:
```py
# Simple example (same image used in the original script)
example_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
labels = ["cat", "dog"]
try:
    res = process("clip", example_url, labels)
    print("CLIP:", res)
except Exception as e:
    print("CLIP run failed:", e)
```
We define the image url, the labels, and then process them with the selected model. At the end of the inference process, we print the probabilities dict.

The resulting output for CLIP:
```
CLIP: {'cat': 0.9971747398376465, 'dog': 0.002825190545991063}
```
CLIP is extremely certain that the image is a cat. And it is correct! Note that CLIP was not directly trained on the MS-COCO dataset, so this prediction shows great embedding power of the model

We can run the same code, but change the `model_name` parameter of the `process` function to `"llava"`. Let's see how it does:
```
LLAVA: {'cat': 0.9556514620780945, 'dog': 0.0443485826253891}
```

LLAVA is 95% certain that the image is a cat. It has a bit smaller certainty, however it is still correctly classifying the image.

Lastly, we can evaluate the performance of the QWEN2 VL model:
```
QWEN: {'cat': 0.9999934434890747, 'dog': 6.540437425428536e-06}
```

QWEN achieves an outstanding accuracy for the given image! But, it is worth noting that QWEN was trained on datasets derived from MS-COCO, so this image could have been included in its training set.

# Conclusion
We have explored the fascinating landscape of how Large Multimodal Models (LMMs) navigate through different types of data using embeddings. We saw how those embeddings are learned, and together we arrived to the same thought process as the LLaVE development team when it came to designing their model. Lastly, we see how fast we've advanced with multimodal models in the last 5 years. On a simple example, different multimodal models were played around with, seeing distinct approaches in the way these models were trained and developed.

## Bibliography
[1] https://arxiv.org/pdf/2503.04812

[2] https://www.alexanderthamm.com/en/blog/an-introduction-to-large-multimodal-models/

[3] https://arxiv.org/abs/2103.00020