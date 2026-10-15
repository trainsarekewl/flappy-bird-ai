import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.flappybird.Bird import Bird
from src.flappybird.Pipes import Pipes
from src.flappybird import config
from src.ai.evolution_ai import evolve
from src.ai.visualize_model import visualize_model, BEST_NET_PATH
import torch

if __name__ == "__main__":
    env = {
        "bird_class": Bird,
        "pipe_class": Pipes,
        "window_height": config.WINDOW_HEIGHT,
        "window_width": config.WINDOW_LENGTH,
    }

    best_net = evolve(env)
    torch.save(best_net.state_dict(), BEST_NET_PATH)

    visualize_model(BEST_NET_PATH)
    print("done")
