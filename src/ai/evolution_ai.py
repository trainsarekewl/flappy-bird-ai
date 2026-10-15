import random

import torch
import torch.nn as nn
import copy
from src.flappybird.difficulty import get_pipe_cooldown, INIT_PIPE_COOLDOWN, MAX_DIFFICULTY_SCORE

# bird y, bird velocity, pipe spawn cooldown, next pipe x/gap offset, second pipe x/gap offset
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

    # gap center relative to the bird: positive means the gap is below the bird
    def gap_offset(gap_center):
        return (gap_center - bird.getYValue()) / window_height

    # a missing pipe is treated as far away (right edge) with a gap centered in the window
    if next_pipe is not None:
        next_x = next_pipe.getXVal() / window_width
        next_height = gap_offset(next_pipe.getHeight() + next_pipe.GAP / 2)
    else:
        next_x, next_height = 1.0, gap_offset(window_height / 2)

    if second_pipe is not None:
        second_x = second_pipe.getXVal() / window_width
        second_height = gap_offset(second_pipe.getHeight() + second_pipe.GAP / 2)
    else:
        second_x, second_height = 1.0, gap_offset(window_height / 2)

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

def avg_score(net, env, seeds):
    return sum(eval_agent(net, env, seed) for seed in seeds) / len(seeds)

# fixed layouts used only to pick the final model, never for selection during training
_validation_rng = random.Random(1234)
VALIDATION_SEEDS = [_validation_rng.randrange(2**32) for _ in range(20)]

def evolve(env, generations=400, pop_size=100, mutation_rate = 0.05, elite_count=3, evals_per_net=5,
           parent_fraction=0.2, min_mutation_rate=0.01):
    nets = [FlappyBirdNet() for _ in range(pop_size)]
    best_net = None
    best_val_score = float("-inf")
    for gen in range(generations):
        # shrink mutations linearly: big jumps early to explore, small tweaks late to fine-tune
        progress = gen / max(generations - 1, 1)
        gen_mutation_rate = mutation_rate + (min_mutation_rate - mutation_rate) * progress

        # several new layouts each generation so fitness rewards consistency, not one lucky run
        gen_seeds = [random.randrange(2**32) for _ in range(evals_per_net)]
        scores = [(avg_score(net, env, gen_seeds), net) for net in nets]
        scores.sort(reverse=True, key=lambda x: x[0])

        # keep the best model ever seen on the validation layouts
        gen_val_score = float("-inf")
        for i in range(elite_count):
            val_score = avg_score(scores[i][1], env, VALIDATION_SEEDS)
            gen_val_score = max(gen_val_score, val_score)
            if val_score > best_val_score:
                best_val_score = val_score
                best_net = copy.deepcopy(scores[i][1])

        print(f"Generation {gen} - Best avg score: {scores[0][0]:.2f} - "
              f"Validation: {gen_val_score:.2f} (best ever {best_val_score:.2f}) - "
              f"Mutation rate: {gen_mutation_rate:.4f}")

        survivors = []

        for i in range(int(pop_size * parent_fraction)):
            survivors.append(scores[i][1])

        # carry the best birds over unmutated so a good bird is never lost
        new_population = [scores[i][1] for i in range(elite_count)]
        for i in range(pop_size - elite_count):
            parent = random.choice(survivors)
            child = copy.deepcopy(parent)
            with torch.no_grad():
                for param in child.parameters():
                    param += torch.randn_like(param) * gen_mutation_rate
            new_population.append(child)
        nets = new_population

    return best_net