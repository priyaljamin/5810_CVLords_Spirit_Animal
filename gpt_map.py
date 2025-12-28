# gpt_map.py - UNLIMITED ANIMAL SELECTION
import os
import base64
import json
from openai import OpenAI
from PIL import Image
from io import BytesIO
from transformers import pipeline

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

SYSTEM = (
    "You are an expert at analyzing facial features and matching them to animals. "
    "Analyze the person's face and choose the BEST matching animal based on their actual features. "
    "Consider face shape, eyes, expression, bone structure, and overall vibe. "
    "You can choose ANY animal in the world - mammals, birds, reptiles, fish, insects, mythical creatures. "
    "Be creative but accurate. Match distinctive features like: "
    "- Wide eyes -> Owl, Lemur, Tarsier "
    "- Round face -> Panda, Koala, Seal "
    "- Angular features -> Wolf, Fox, Hawk "
    "- Strong jaw -> Lion, Tiger, Bear "
    "- Playful expression -> Otter, Dolphin, Monkey "
    "- Calm demeanor -> Sloth, Turtle, Elephant "
    "- Unique features -> Chameleon, Octopus, Peacock "
    "Return STRICT JSON with keys: animal, reason, prompt, negative."
)


def _img_to_data_url(path):
    """Convert image file to base64 data URL"""
    with Image.open(path) as img:
        img = img.convert("RGB")
        img.thumbnail((512, 512))
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def suggest_animal(img_path: str) -> dict:
    data_url = _img_to_data_url(img_path)

    user_prompt = (
        "Analyze this face and choose the BEST matching animal from the ENTIRE animal kingdom. "
        "Match based on actual facial features, not stereotypes. "
        "Return JSON ONLY with this format: "
        '{"animal":"[animal name]","reason":"[specific facial features that match]",'
        '"prompt":"replace the head with a [animal] head, same head angle and expression, '
        'photorealistic, detailed features, natural look, anatomically correct",'
        '"negative":"human face, human head, person, portrait, extra limbs, deformed, text, watermark"}'
    )

    models = ["gpt-4o-mini", "gpt-4o"]
    data = None

    for model in models:
        try:
            print(f"🔹 Trying {model}...")
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": data_url}}
                    ]}
                ],
                temperature=0.3,
                max_tokens=300,
                response_format={"type": "json_object"}
            )

            raw_content = resp.choices[0].message.content
            data = json.loads(raw_content)
            print(f"✅ Success with {model}")
            break

        except Exception as e:
            print(f"❌ Error with {model}: {e}")
            continue

    # Free local fallback
    if not data:
        print("⚠ GPT models failed - trying CLIP fallback...")
        try:
            vision = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")
            candidate_animals = [
                "lion", "tiger", "wolf", "fox", "owl", "eagle", "bear", "deer", "dog", "cat",
                "elephant", "giraffe", "horse", "dolphin", "otter", "panda", "sloth", "monkey"
            ]
            result = vision(img_path, candidate_labels=candidate_animals)
            best = max(result, key=lambda x: x["score"])
            animal = best["label"]

            data = {
                "animal": animal,
                "reason": f"CLIP model matched to {animal} (confidence {best['score']:.2f})",
                "prompt": f"replace the head with a {animal} head, same head angle, photorealistic",
                "negative": "human face, person, text, watermark"
            }
            print(f"✅ CLIP fallback chose: {animal}")
        except Exception as e:
            print(f"❌ CLIP fallback failed: {e}")

    # Static fallback
    if not data:
        data = {
            "animal": "owl",
            "reason": "fallback due to API issues",
            "prompt": "replace the head with an owl head, photorealistic, same pose",
            "negative": "human face, person, text, watermark"
        }

    return data