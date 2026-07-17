import sys
from ddgs import DDGS
import requests
import os
import time
import re
from PIL import Image
import io

MAX_IMAGES = 1000
MAX_RESULTS_PER_KEYWORD = 300
DELAY = 0.3

BASE_EXCLUDE_KEYWORDS = [
    "cartoon", "anime", "illustration", "drawing", "clipart",
    "vector", "painting", "art", "generated", "ai", "midjourney",
    "stable diffusion", "dall-e", "digital art", "3d", "render",
    "pixel", "sketch", "manga", "comic",
    "shutterstock", "gettyimages", "alamy", "dreamstime",
    "istockphoto", "depositphotos", "123rf", "adobe stock",
]

CATEGORIES = {
    "cats": {
        "save_dir": r"PATH",
        "prefix": "cat",
        "exclude_keywords": BASE_EXCLUDE_KEYWORDS,
        "keywords": [
            "real cat photo",
            "cat photograph realistic",
            "domestic cat real photo",
            "cat portrait photography",
            "cat wildlife photo",
            "cute cat real picture",
            "cat close up photo",
            "kitten real photo",
            "tabby cat photo",
            "cat outdoor photo",
            "persian cat photo",
            "british shorthair cat photo",
            "cat sitting photo",
            "cat indoor photo",
            "fluffy cat photo",
            "maine coon cat photo",
            "siamese cat photo",
            "bengal cat photo",
            "ragdoll cat photo",
            "sphynx cat photo",
            "black cat real photo",
            "white cat real photo",
            "orange cat real photo",
            "cat sleeping photo",
            "cat playing photo",
            "street cat photo",
            "cat on grass photo",
            "cat face close up photo",
            "two cats photo",
            "cat running photo",
            "scottish fold cat photo",
            "abyssinian cat photo",
            "norwegian forest cat photo",
            "calico cat real photo",
            "grey cat real photo",
            "ginger cat photo",
            "cat lying down photo",
            "cat on sofa photo",
            "cat in garden photo",
            "cat looking at camera photo",
            "stray cat photo",
            "cat stretching photo",
        ],
    },
    "dogs": {
        "save_dir": r"PATH",
        "prefix": "dog",
        "exclude_keywords": BASE_EXCLUDE_KEYWORDS,
        "keywords": [
            "real dog photo",
            "dog photograph realistic",
            "domestic dog real photo",
            "dog portrait photography",
            "dog outdoor photo",
            "cute dog real picture",
            "dog close up photo",
            "puppy real photo",
            "golden retriever photo",
            "labrador dog photo",
            "german shepherd photo",
            "bulldog real photo",
            "dog sitting photo",
            "dog indoor photo",
            "fluffy dog photo",
            "poodle dog photo",
            "beagle dog photo",
            "husky dog photo",
            "chihuahua dog photo",
            "rottweiler dog photo",
            "border collie dog photo",
            "dog running photo",
            "dog playing photo",
            "dog on grass photo",
            "black dog real photo",
            "white dog real photo",
            "brown dog real photo",
            "dog face close up photo",
            "two dogs photo",
            "small dog photo",
            "dalmatian dog photo",
            "boxer dog photo",
            "shiba inu dog photo",
            "pug dog photo",
            "corgi dog photo",
            "great dane dog photo",
            "dog lying down photo",
            "dog on sofa photo",
            "dog in garden photo",
            "dog looking at camera photo",
            "dog at the beach photo",
            "dog in snow photo",
        ],
    },
    "others": {
        "save_dir": r"PATH",
        "prefix": "other",
        "exclude_keywords": BASE_EXCLUDE_KEYWORDS + ["cat", "cats", "kitten", "dog", "dogs", "puppy"],
        "keywords": [
            # Nature / paysages
            "mountain landscape photo",
            "beach sunset photo",
            "forest nature photo",
            "waterfall photo",
            "desert landscape photo",
            "snow mountain photo",
            "lake reflection photo",
            # Nourriture
            "pizza real photo",
            "burger food photo",
            "sushi real photo",
            "fruits basket photo",
            "coffee cup photo",
            "pasta dish photo",
            "salad bowl photo",
            # Objets du quotidien
            "car real photo",
            "motorcycle photo",
            "bicycle photo",
            "chair furniture photo",
            "smartphone photo",
            "laptop computer photo",
            "watch wrist photo",
            # Sport
            "football match photo",
            "basketball player photo",
            "tennis court photo",
            "swimming pool photo",
            # Animaux (non chat/chien)
            "rabbit real photo",
            "horse real photo",
            "bird real photo",
            "lion real photo",
            "elephant real photo",
            "cow farm photo",
            "fish underwater photo",
            # Personnes / lieux
            "city street photo",
            "market place photo",
            "office desk photo",
            "gym workout photo",
            "concert crowd photo",
            "airport terminal photo",
            "library books photo",
            # encore plus de variété
            "river valley photo",
            "autumn forest photo",
            "bread bakery photo",
            "ice cream photo",
            "truck vehicle photo",
            "train station photo",
            "guitar instrument photo",
            "soccer stadium photo",
            "deer wildlife photo",
            "monkey real photo",
            "village houses photo",
            "kitchen interior photo",
        ],
    },
}


# verifier si l'image est valide et n'est corrumpu
def is_valid_image(content: bytes) -> bool:
    try:
        img = Image.open(io.BytesIO(content))
        img.verify()
        return True
    except Exception:
        return False

# verifier si l'image n'est pas AI ou 3D
def is_realistic(result: dict, exclude_keywords: list) -> bool:
    text_to_check = " ".join([
        result.get("title", ""),
        result.get("image", ""),
        result.get("url", ""),
        result.get("source", ""),
    ]).lower()

    for word in exclude_keywords:
        # verifier le regex (faut qu'il soit un mot seule '\bart\b')
        if re.search(r"\b" + re.escape(word) + r"\b", text_to_check):
            return False
    return True


def download_images(keywords, exclude_keywords, prefix, max_images, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    saved = 0
    skipped = 0
    keyword_index = 0
    seen_urls = set()

    while saved < max_images:
        if keyword_index >= len(keywords):
            break

        keyword = keywords[keyword_index]
        keyword_index += 1

        # recuperes les metadonnées (dict: {titles, images, url, source})
        with DDGS() as ddgs:
            try:
                results = list(ddgs.images(
                    keyword,
                    max_results=MAX_RESULTS_PER_KEYWORD,
                    type_image="photo",
                ))
            except Exception:
                time.sleep(5)
                continue

        for r in results:
            if saved >= max_images:
                break

            img_url = r.get("image", "")
            if img_url in seen_urls:
                continue
            seen_urls.add(img_url)

            if not is_realistic(r, exclude_keywords):
                skipped += 1
                continue

            try:
                response = requests.get(img_url, timeout=6)

                if not is_valid_image(response.content):
                    skipped += 1
                    continue

                img = Image.open(io.BytesIO(response.content)).convert("RGB")

                w, h = img.size
                if w < 150 or h < 150:
                    skipped += 1
                    continue

                file_path = os.path.join(save_dir, f"{prefix}_{saved + 1:04d}.jpg")
                img.save(file_path, "JPEG", quality=90)

                saved += 1
                time.sleep(DELAY)

            except Exception:
                skipped += 1

    print(f"\n{prefix} : {saved} images sauvegardées, {skipped} ignorées.")


if __name__ == "__main__":
    categories = sys.argv[1:] or list(CATEGORIES.keys())

    for name in categories:
        cat = CATEGORIES[name]
        download_images(
            cat["keywords"],
            cat["exclude_keywords"],
            cat["prefix"],
            MAX_IMAGES,
            cat["save_dir"],
        )
