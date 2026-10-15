import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ai.visualize_model import visualize_model, BEST_NET_PATH

if __name__ == "__main__":
    visualize_model(BEST_NET_PATH)
