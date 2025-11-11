import random

import torch
import torch.nn as nn
import copy

class FlappyBirdNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)

def eval_agent(net, env, max_frames=3000):
    INIT_PIPE_COOLDOWN = env["init_pipe_cooldown"]

    score = 0
    bird = env["bird_class"](50, 150)
    obstacles = []
    pipe_cd = INIT_PIPE_COOLDOWN
    last_pipe_time = pipe_cd

    for frame in range(max_frames):
        if last_pipe_time >= pipe_cd:
            obstacles.append(env["pipe_class"]())
            last_pipe_time = 0
        last_pipe_time += 1

        next_pipe = None
        for pipe in obstacles:
            if pipe.getXVal() + pipe.getWidth() > bird.getXValue():
                next_pipe = pipe
                break

        if next_pipe is not None:
            inputs = torch.tensor([
                bird.getYValue() / env["window_height"],
                bird.getYVelocity() / 10,
                next_pipe.getXVal() / env["window_width"],
                next_pipe.getHeight() /env["window_height"]
            ], dtype=torch.float32)
        else:
            inputs = torch.tensor([
                bird.getYValue() / env["window_height"],
                bird.getYVelocity() / 10,
                0,
                0
            ])

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
            return score + frame * 0.01
    return score + frame * 0.01

def evolve(env, generations=40, pop_size=100, mutation_rate = 0.1):
    nets = [FlappyBirdNet() for _ in range(pop_size)]
    for gen in range(generations):
        scores = [(eval_agent(net, env), net) for net in nets]
        scores.sort(reverse=True, key=lambda x: x[0])
        print(f"Generation {gen} - Best score: {scores[0][0]}")

        survivors = []

        for i in range(int(pop_size * 0.1)):
            survivors.append(scores[i][1])

        new_population = []
        for i in range(pop_size):
            parent = random.choice(survivors)
            child = copy.deepcopy(parent)
            with torch.no_grad():
                for param in child.parameters():
                    param += torch.randn_like(param) * mutation_rate
            new_population.append(child)
        nets = new_population

    return scores[0][1]