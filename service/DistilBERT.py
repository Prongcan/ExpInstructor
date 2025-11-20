from transformers import pipeline
import torch
import json
import os

def build_novelty_classifier(model_path="service/BERT_model"):
    """Build the novelty classifier with GPU->CPU fallback"""
    
    if not os.path.exists(model_path):
        print(f"Error: model path does not exist: {model_path}")
        print("Please run finetune_novelty_classifier.py to train the model first")
        return None, None
    
    # Load label mapping
    label_mapping_file = os.path.join(model_path, "label_mapping.json")
    if os.path.exists(label_mapping_file):
        with open(label_mapping_file, 'r') as f:
            label_mapping = json.load(f)
        id2label = label_mapping["id2label"]
        print("Label mapping:")
        for label_id, label_name in id2label.items():
            print(f"  {label_id} -> {label_name}")
    else:
        print("Warning: label mapping file not found")
        id2label = None
    
    # Try GPU
    if torch.cuda.is_available():
        try:
            clf = pipeline(
                task="text-classification",
                model=model_path,
                torch_dtype=torch.float16,
                device=0,
            )
            print("Using GPU: cuda:0 (fp16)")
            return clf, "cuda:0"
        except Exception as gpu_error:
            print(f"GPU initialization failed, falling back to CPU. Reason: {gpu_error}")
    
    # CPU fallback
    clf = pipeline(
        task="text-classification",
        model=model_path,
        torch_dtype=torch.float32,
        device=-1,
    )
    print("Using CPU (fp32)")
    return clf, "cpu"

def predict_novelty_score(classifier, text):
    """Predict the novelty score"""
    if classifier is None:
        return None
    
    try:
        result = classifier(text)
        
        # Process the result
        if isinstance(result, list) and len(result) > 0:
            prediction = result[0]
            label = prediction['label']
            score = prediction['score']
            
            # Convert the label ID to an actual score (1-10)
            try:
                predicted_score = int(label)
            except ValueError:
                predicted_score = label
            
            return {
                'predicted_score': predicted_score,
                'confidence': score,
                'raw_result': result
            }
        else:
            return None
            
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

def main():
    """Main entry point"""
    print("=== Novelty Classifier Inference ===")
    
    # Build the classifier
    classifier, device = build_novelty_classifier()
    
    if classifier is None:
        return
    
    # Interactive inference
    print("\nType 'quit' to exit")
    print("Enter text and the model will predict a novelty score (1-10)")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\nEnter text: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Exiting program")
                break
            
            if not user_input:
                print("Please enter valid text")
                continue
            
            # Prediction
            result = predict_novelty_score(classifier, user_input)
            
            if result:
                print(f"\nPrediction:")
                print(f"  Novelty score: {result['predicted_score']}/10")
                print(f"  Confidence: {result['confidence']:.4f}")
            else:
                print("Prediction failed")
                
        except KeyboardInterrupt:
            print("\n\nProgram interrupted")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
