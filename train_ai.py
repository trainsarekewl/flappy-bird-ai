from src.flappybird.Bird import Bird
from src.flappybird.Pipes import Pipes
from src.flappybird import config
from src.ai.evolution_ai import evolve
from src.ai.visualize_model import visualize_model
import torch

env = {
    "bird_class": Bird,
    "pipe_class": Pipes,
    "window_height": config.WINDOW_HEIGHT,
    "window_width": config.WINDOW_LENGTH,
}

best_net = evolve(env)
torch.save(best_net.state_dict(), "best_net.pt")

visualize_model()
print("done")