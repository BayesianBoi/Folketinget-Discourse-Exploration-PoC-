import sys
import os
from pathlib import Path
import joblib
import pandas as pd
from turftopic.analyzers import OpenAIAnalyzer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dk_politics_topics import DEFAULT_CONFIG
from dk_politics_topics.modeling import label_topics
from dk_politics_topics.utils.cache import save_json

def main():
    print("Checking API Key...")
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not found in environment.")
        return

    model_path = DEFAULT_CONFIG.paths.models_dir / "topeax_model.joblib"
    print(f"Loading model from {model_path}...")
    wrapper = joblib.load(model_path)
    
    # Extract internal model
    if hasattr(wrapper, "model"):
        model = wrapper.model
    else:
        model = wrapper
        
    print("Renaming topics with OpenAI...")
    try:
        analyzer = OpenAIAnalyzer(model_name=DEFAULT_CONFIG.topeax.openai_model)
        model.rename_topics(analyzer)
        print("Success! New topic names:")
        print(model.topic_names)
        
        # Save the updated model (wrapper) back
        # We need to make sure the wrapper references the updated model if it was separate
        # But usually wrapper.model *is* the object we modified.
        joblib.dump(wrapper, model_path)
        print("Saved updated model.")
        
        # Also update topics.json
        topics_path = DEFAULT_CONFIG.paths.exports_dir / DEFAULT_CONFIG.export.topic_json
        import json
        payload = json.loads(topics_path.read_text(encoding="utf-8"))
        
        labels = {}
        for idx, name in enumerate(model.topic_names):
            labels[str(idx)] = name
            
        payload["metadata"]["labels"] = labels
        save_json(payload, topics_path)
        print("Updated topics.json labels.")
        
    except Exception as e:
        print(f"Error renaming topics: {e}")

if __name__ == "__main__":
    main()
