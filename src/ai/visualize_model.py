import pygame
import torch
from src.flappybird import config
from src.flappybird.Bird import Bird
from src.flappybird.Pipes import Pipes
from src.ai.evolution_ai import evolve, FlappyBirdNet


def visualize_model(model_path = "best_net.pt"):
    net = FlappyBirdNet()
    net.load_state_dict(torch.load(model_path))

    pygame.init()
    window = pygame.display.set_mode((config.WINDOW_LENGTH, config.WINDOW_HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Comic Sans MS", 20)

    bird = Bird(50, 150)
    obstacles = []
    init_pipe_cooldown = (int) (config.WINDOW_LENGTH / 9)
    pipe_cd = init_pipe_cooldown
    last_pipe_time = pipe_cd
    score = 0
    game = True
    PIPE_COOLDOWN_CHANGE_RATE = config.PIPE_COOLDOWN_CHANGE_RATE
    last_cooldown_update = 0

    while game:
        clock.tick(60)
        window.fill(config.COLOR)

        # handle quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game = False

        if score % 2 == 0 and score != 0 and score != last_cooldown_update and pipe_cd > 45:
            pipe_cd -= PIPE_COOLDOWN_CHANGE_RATE
            last_cooldown_update = score

        if last_pipe_time >= pipe_cd:
            obstacles.append(Pipes())
            last_pipe_time = 0
        last_pipe_time += 1

        next_pipe = None
        for pipe in obstacles:
            if pipe.getXVal() + pipe.getWidth() > bird.getXValue():
                next_pipe = pipe
                break

        if next_pipe is not None:
            inputs = torch.tensor([
                bird.getYValue() / config.WINDOW_HEIGHT,
                bird.getYVelocity() / 10,
                next_pipe.getXVal() / config.WINDOW_LENGTH,
                next_pipe.getHeight() / config.WINDOW_HEIGHT
            ], dtype=torch.float32)
        else:
            inputs = torch.tensor([
                bird.getYValue() / config.WINDOW_HEIGHT,
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
            pipe.draw(window)

            topRect = pipe.getTopRect()
            bottomRect = pipe.getBottomRect()
            birdRect = bird.getRect()

            # died/lost
            if birdRect.colliderect(bottomRect) or birdRect.colliderect(topRect):
                game = False

        removed = 0
        for i in removePipe:
            obstacles.pop(i - removed)
            score += 1
            removed += 1

        if bird.getYValue() < 0 or bird.getYValue() >= config.WINDOW_HEIGHT - 10:
            game = False

        # draw stuff
        stringScore = str(score)
        StringPipeCD = str(pipe_cd)
        textScore = font.render("score: " + stringScore, 1, (125, 125, 125))
        textPipeCD = font.render("Pipe CD: " + StringPipeCD, 1, (125, 125, 125))
        window.blit(textScore, (0, 0))
        window.blit(textPipeCD, (0, 20))

        bird.draw(window)
        pygame.display.update()


    pygame.quit()
    print(score)

