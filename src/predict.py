import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# OPTIMIZATION: Force single thread to match latency script
torch.set_num_threads(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    # 1. Load Model & Tokenizer
    print(f"Loading model from {args.model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    
    # 2. APPLY QUANTIZATION (The Speed Hack)
    # Note: The warning you saw earlier is fine, we can ignore it for this assignment.
    print("Quantizing model...")
    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    model.eval()
    
    id2label = model.config.id2label
    
    # CHANGE: Use a dictionary instead of a list
    predictions = {}
    
    with open(args.input, "r") as f:
        data = [json.loads(line) for line in f]

    print(f"Predicting on {len(data)} examples...")

    for item in data:
        text = item["text"]
        
        # 3. FAST TOKENIZATION (Truncation=True, max_length=64)
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=64, 
            return_offsets_mapping=True
        )
        
        offset_mapping = inputs.pop("offset_mapping")[0]
        
        with torch.no_grad():
            logits = model(**inputs).logits
        
        pred_ids = torch.argmax(logits, dim=2)[0].tolist()
        
        entities = []
        current_entity = None
        
        for idx, pred_id in enumerate(pred_ids):
            label_name = id2label[pred_id]
            
            if idx >= len(offset_mapping) or (offset_mapping[idx][0] == 0 and offset_mapping[idx][1] == 0):
                continue
                
            if label_name.startswith("B-"):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    "label": label_name[2:],
                    "start": offset_mapping[idx][0].item(),
                    "end": offset_mapping[idx][1].item()
                }
            elif label_name.startswith("I-"):
                if current_entity and current_entity["label"] == label_name[2:]:
                    current_entity["end"] = offset_mapping[idx][1].item()
                else:
                    if current_entity:
                        entities.append(current_entity)
                    current_entity = {
                        "label": label_name[2:],
                        "start": offset_mapping[idx][0].item(),
                        "end": offset_mapping[idx][1].item()
                    }
            else: 
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        if current_entity:
            entities.append(current_entity)
            
        # CHANGE: Save to dictionary using the ID as the key
        predictions[item["id"]] = entities

    # Save
    with open(args.output, "w") as f:
        json.dump(predictions, f, indent=2)
        
    print(f"Saved predictions to {args.output}")

if __name__ == "__main__":
    main()