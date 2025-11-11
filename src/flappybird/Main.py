import pygame
import sys
from pathlib import Path
from src.flappybird.Bird import Bird
from Pipes import Pipes
from src.flappybird import config

BASE_DIR = Path(__file__).resolve().parent.parent

# settings
WINDOW_HEIGHT = config.WINDOW_HEIGHT
WINDOW_LENGTH = config.WINDOW_LENGTH
JUMP_COOLDOWN = config.JUMP_COOLDOWN
PIPE_SPEED = config.PIPE_SPEED
COLOR = config.COLOR
PIPE_COOLDOWN_CHANGE_RATE = config.PIPE_COOLDOWN_CHANGE_RATE

def main():
    # initalize pygame stuff
    pygame.init()
    pygame.font.init()
    clock = pygame.time.Clock()

    # settings
    BIRD_HEIGHT = 150
    INIT_PIPE_COOLDOWN = (int) (WINDOW_LENGTH / 9)


    # initalize variables
    cooldown = 0
    game = False
    obstacles = []
    pipe_cool = 0
    jumping = True
    score = 0
    scoreFont = pygame.font.SysFont("Comic Sans MS", 20)
    pipe_cooldown = INIT_PIPE_COOLDOWN
    last_cooldown_update = 0


    pygame.display.set_caption("flappy bird")

    # create bird
    bird = Bird(50, BIRD_HEIGHT)

    # create window
    window = pygame.display.set_mode((WINDOW_LENGTH, WINDOW_HEIGHT))

    while True:
        # update screen and make 60 fps
        clock.tick(60)
        window.fill(COLOR)

        # every 10 score decrease pipe cooldown
        if score % 2 == 0 and score != 0 and score != last_cooldown_update and pipe_cooldown > 45:
            pipe_cooldown -= PIPE_COOLDOWN_CHANGE_RATE
            last_cooldown_update = score

        # check events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                # jump
                if event.key == pygame.K_SPACE and cooldown == 0 and jumping == True:
                    if game == False:
                        bird.jump()
                        game = True
                        cooldown = JUMP_COOLDOWN

                    else:
                        bird.jump()
                        cooldown = JUMP_COOLDOWN

                # restart
                if event.key == pygame.K_r:
                    jumping = True
                    bird.setYValue(BIRD_HEIGHT)
                    score = 0
                    pipe_cooldown = INIT_PIPE_COOLDOWN
                    cooldown = 0
                    for i in range(0, len(obstacles)):
                        obstacles.pop()



        # pipe stuff
        # add pipe
        if game and pipe_cool <= 0:
            pipes = Pipes()
            obstacles.append(pipes)
            pipe_cool = pipe_cooldown

        pipe_cool -= 1

        # iterate through all pipes
        removePipe = []
        indx = 0
        for pipe in obstacles:
            # check if out of bounds
            if pipe.getXVal() < - pipe.getWidth():
                removePipe.append(indx)
            # draw and move the pipe
            if game and jumping:
                pipe.move()

            pipe.draw(window)

            # check collision and make it so bird cant jump and pipes stop moving
            topRect = pipe.getTopRect()
            bottomRect = pipe.getBottomRect()
            birdRect = bird.getRect()

            if birdRect.colliderect(bottomRect) or birdRect.colliderect(topRect):
                jumping = False
            indx += 1

        # remove pipe
        removed = 0
        for indx in removePipe:
            obstacles.pop(indx - removed)
            score += 1
            removed += 1

        # bird stuff

        # check if bird is is in game
        if bird.getYValue() < 0:
            bird.setYValue(0)

        if bird.getYValue() >= WINDOW_HEIGHT - 10:
            bird.setYValue(WINDOW_HEIGHT - 10)
            gameOverFont = pygame.font.SysFont("Comic Sans MS", 50)
            text_surface = gameOverFont.render('GAME OVER', False, (255, 0, 0))
            window.blit(text_surface, (WINDOW_LENGTH / 3, WINDOW_HEIGHT / 2.5))
            jumping = False
            game = False

        # check if game has started conditions
        if game and bird.getYValue() < WINDOW_HEIGHT - 10:
            bird.fall()

        # draw stuff
        bird.draw(window)

        # display font
        stringScore = str(score)
        textScore = scoreFont.render("score: " + stringScore, 1, (125, 125, 125))
        window.blit(textScore, (0, 0))

        pygame.display.update()

        if cooldown > 0 :
            cooldown -= 1


main()
