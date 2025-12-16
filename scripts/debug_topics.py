import sys
from pathlib import Path
import joblib
from pprint import pprint

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
    
from dk_politics_topics import DEFAULT_CONFIG

def debug():
    model_path = DEFAULT_CONFIG.paths.models_dir / "topeax_model.joblib"
    print(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    
    print("Type of model:", type(model))
    
    if hasattr(model, "get_topics"):
        print("Calling get_topics()...")
        try:
            topics = model.get_topics(top_k=5)
        except TypeError:
            print("get_topics(top_k=5) failed, trying no args")
            topics = model.get_topics()
            
        print("Type of topics:", type(topics))
        if isinstance(topics, dict):
            print("Keys:", list(topics.keys())[:5])
            print("First topic value:")
            pprint(topics[next(iter(topics))])
        elif isinstance(topics, list):
            print("First topic value:")
            pprint(topics[0])
    else:
        print("Model has no get_topics method")

if __name__ == "__main__":
    debug()
