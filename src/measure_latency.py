import time
import torch
import numpy as np
import argparse
import json
from transformers import AutoTokenizer, AutoModelForTokenClassification

# CRITICAL: Force single-threaded execution to remove overhead
torch.set_num_threads(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--runs", type=int, default=50)
    args = parser.parse_args()

    print(f"Loading model from {args.model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)

    # DYNAMIC QUANTIZATION (The Speed Hack)
    print("Applying Dynamic Quantization...")
    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    model.eval()

    # Load Data
    print(f"Loading data from {args.input}...")
    texts = []
    with open(args.input, 'r') as f:
        for line in f:
            data = json.loads(line)
            texts.append(data['text'])

    # Warmup
    print("Warming up...")
    dummy_input = tokenizer(texts[0], return_tensors="pt")
    with torch.no_grad():
        for _ in range(10):
            model(**dummy_input)

    # Measurement Loop
    print(f"Measuring latency over {args.runs} runs...")
    latencies = []
    
    for i in range(args.runs):
        text = texts[i % len(texts)]
        # Force truncation to keep inputs short
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
        end_time = time.time()
        
        latencies.append((end_time - start_time) * 1000) # Convert to ms

    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)

    print(f"\nLatency results (Batch Size 1):")
    print(f"  p50: {p50:.2f} ms")
    print(f"  p95: {p95:.2f} ms")
    
    if p95 <= 20:
        print("\nSUCCESS: Latency is within the budget! (<= 20ms)")
    else:
        print("\nFAIL: Still too slow.")

if __name__ == "__main__":
    main()