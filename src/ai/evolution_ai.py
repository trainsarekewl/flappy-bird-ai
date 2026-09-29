import random

import torch
import torch.nn as nn
import copy
from src.flappybird.difficulty import get_pipe_cooldown, INIT_PIPE_COOLDOWN, MAX_DIFFICULTY_SCORE

# bird y, bird velocity, pipe spawn cooldown, next pipe x/height, second pipe x/height
NUM_INPUTS = 7
# scale down the second pipe's inputs so they don't drown out the next pipe
SECOND_PIPE_WEIGHT = 0.5

class FlappyBirdNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(NUM_INPUTS, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)

def build_inputs(bird, obstacles, window_width, window_height, pipe_cd):
    upcoming = [pipe for pipe in obstacles if pipe.getXVal() + pipe.getWidth() > bird.getXValue()]
    next_pipe = upcoming[0] if len(upcoming) > 0 else None
    second_pipe = upcoming[1] if len(upcoming) > 1 else None

    # a missing pipe is treated as far away (right edge) with a centered gap
    if next_pipe is not None:
        next_x = next_pipe.getXVal() / window_width
        next_height = next_pipe.getHeight() / window_height
    else:
        next_x, next_height = 1.0, 0.5

    if second_pipe is not None:
        second_x = second_pipe.getXVal() / window_width
        second_height = second_pipe.getHeight() / window_height
    else:
        second_x, second_height = 1.0, 0.5

    return torch.tensor([
        bird.getYValue() / window_height,
        bird.getYVelocity() / 20,
        pipe_cd / INIT_PIPE_COOLDOWN,
        next_x,
        next_height,
        second_x * SECOND_PIPE_WEIGHT,
        second_height * SECOND_PIPE_WEIGHT,
    ], dtype=torch.float32)

# run past the hardest difficulty so the birds actually train on it
MAX_SCORE = MAX_DIFFICULTY_SCORE + 50

def eval_agent(net, env, seed=None, max_score=MAX_SCORE, max_frames=50000):
    pipe_rng = random.Random(seed)
    score = 0
    bird = env["bird_class"](50, 150)
    obstacles = []
    last_pipe_time = INIT_PIPE_COOLDOWN

    for frame in range(max_frames):
        if score >= max_score:
            break

        pipe_cd = get_pipe_cooldown(score)

        if last_pipe_time >= pipe_cd:
            obstacles.append(env["pipe_class"](pipe_rng))
            last_pipe_time = 0
        last_pipe_time += 1

        inputs = build_inputs(bird, obstacles, env["window_width"], env["window_height"], pipe_cd)

        if net(inputs).item() > 0.5:
            bird.jump()
        bird.fall()

        removePipe = []
        for i, pipe in enumerate(obstacles):
            if pipe.getXVal() < - pipe.getWidth():
                removePipe.append(i)
            pipe.move()

            topRect = pipe.getTopRect()
            bottomRect = pipe.getBottomRect()
            birdRect = bird.getRect()

            # died/lost
            if birdRect.colliderect(bottomRect) or birdRect.colliderect(topRect):
                return score + frame * 0.01

        removed = 0
        for i in removePipe:
            obstacles.pop(i - removed)
            score += 1
            removed += 1

        if bird.getYValue() < 0 or bird.getYValue() >= env["window_height"] - 10:
            return score + frame * 0.01 - 5
    return score + frame * 0.01

def evolve(env, generations=200, pop_size=150, mutation_rate = 0.05, elite_count=3):
    nets = [FlappyBirdNet() for _ in range(pop_size)]
    for gen in range(generations):
        # new layout each generation so the birds don't memorize one set of pipes
        gen_seed = random.randrange(2**32)
        scores = [(eval_agent(net, env, gen_seed), net) for net in nets]
        scores.sort(reverse=True, key=lambda x: x[0])
        print(f"Generation {gen} - Best score: {scores[0][0]}")

        survivors = []

        for i in range(int(pop_size * 0.1)):
            survivors.append(scores[i][1])

        # carry the best birds over unmutated so a good bird is never lost
        new_population = [scores[i][1] for i in range(elite_count)]
        for i in range(pop_size - elite_count):
            parent = random.choice(survivors)
            child = copy.deepcopy(parent)
            with torch.no_grad():
                for param in child.parameters():
                    param += torch.randn_like(param) * mutation_rate
            new_population.append(child)
        nets = new_population

    return scores[0][1]