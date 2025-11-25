import json
import random
from faker import Faker

fake = Faker()

# Config
NUM_TRAIN = 800
NUM_DEV = 200
OUTPUT_DIR = "data"

# Labels as per prompt [cite: 8]
LABELS = ["CREDIT_CARD", "PHONE", "EMAIL", "PERSON_NAME", "DATE", "CITY", "LOCATION"]

def text_to_noisy_stt(text):
    """
    Simulates STT noise.
    1. Lowercase everything.
    2. numbers -> words (occasionally).
    3. symbols -> words (@ -> at, . -> dot).
    """
    # Basic mapping for noise
    replacements = {
        ".": " dot ",
        "@": " at ",
        "-": " ",
        ",": "",
        "?": "",
        "!": ""
    }
    
    noisy_text = text.lower()
    for k, v in replacements.items():
        noisy_text = noisy_text.replace(k, v)
        
    # Note: A real production system would need a num2words library here
    # to convert "123" to "one two three". For this script, we assume
    # basic digit retention or simple mapping for speed.
    
    # Clean up double spaces created by replacements
    return " ".join(noisy_text.split())

def generate_sample(id_num):
    """Generates a single labeled example."""
    data = {
        "id": f"utt_{id_num:04d}",
        "text": "",
        "entities": []
    }
    
    # 1. Create a base sentence template
    templates = [
        "my name is {PERSON_NAME}",
        "contact me at {EMAIL}",
        "call me on {PHONE}",
        "i live in {CITY}",
        "meeting is on {DATE}",
        "card number is {CREDIT_CARD}",
        "we are located at {LOCATION}",
        "hi this is {PERSON_NAME} from {CITY} my email is {EMAIL}"
    ]
    
    template = random.choice(templates)
    
    # 2. Generate fake PII
    pii_map = {
        "{PERSON_NAME}": fake.name(),
        "{EMAIL}": fake.email(),
        "{PHONE}": fake.phone_number(),
        "{CITY}": fake.city(),
        "{DATE}": fake.date(),
        "{CREDIT_CARD}": fake.credit_card_number(),
        "{LOCATION}": fake.address()
    }
    
    # 3. Construct the string and track offsets
    # We split the template to insert PII and track their positions
    parts = template.split()
    final_tokens = []
    entities = []
    
    current_char_idx = 0
    
    for part in parts:
        clean_part = part # default
        label = None
        
        # Check if this part is a placeholder
        if part in pii_map:
            raw_value = pii_map[part]
            # APPLY NOISE TO PII VALUE
            noisy_value = text_to_noisy_stt(raw_value)
            clean_part = noisy_value
            
            # Determine Label
            label = part.strip("{}")
            
        else:
            # Apply noise to filler words
            clean_part = text_to_noisy_stt(part)

        # Add to final text
        # Handle spacing
        prefix = "" if current_char_idx == 0 else " "
        segment = prefix + clean_part
        
        start_idx = current_char_idx + len(prefix)
        end_idx = start_idx + len(clean_part)
        
        if label:
            entities.append({
                "start": start_idx,
                "end": end_idx,
                "label": label
            })
            
        final_tokens.append(clean_part)
        current_char_idx = end_idx

    data["text"] = " ".join(final_tokens)
    data["entities"] = entities
    return data

# Generate Files
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Generating {NUM_TRAIN} train examples...")
with open(f"{OUTPUT_DIR}/train.jsonl", "w") as f:
    for i in range(NUM_TRAIN):
        f.write(json.dumps(generate_sample(i)) + "\n")

print(f"Generating {NUM_DEV} dev examples...")
with open(f"{OUTPUT_DIR}/dev.jsonl", "w") as f:
    for i in range(NUM_DEV):
        f.write(json.dumps(generate_sample(i)) + "\n")

print("Done! Data generated in data/ folder.")