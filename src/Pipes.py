import pygame
import random
from pathlib import Path
import config

WINDOW_HEIGHT = config.WINDOW_HEIGHT
WINDOW_LENGTH = config.WINDOW_LENGTH
JUMP_COOLDOWN = config.JUMP_COOLDOWN
PIPE_SPEED = config.PIPE_SPEED
COLOR = config.COLOR
BASE_DIR = config.BASE_DIR

class Pipes(pygame.sprite.Sprite):
    BASE_DIR = Path(__file__).resolve().parent.parent

    # class variables
    COLOR = (0, 0, 0)
    GAP = 140
    WIDTH = 50
    HEIGHT = 0

    def __init__(self):
        self.HEIGHT = random.randrange(10, WINDOW_HEIGHT - self.GAP + 10)
        self.rectTop = pygame.Rect(WINDOW_LENGTH - 5, 0, self.WIDTH, self.HEIGHT)
        self.rectBottom = pygame.Rect(WINDOW_LENGTH - 5, 0 + self.HEIGHT + self.GAP, self.WIDTH, 480 - self.HEIGHT)
        self.mask = None

    def draw(self, window):
        pygame.draw.rect(window, self.COLOR, self.rectTop)
        pygame.draw.rect(window, self.COLOR, self.rectBottom)

    def move(self):
        self.rectTop.x -= PIPE_SPEED
        self.rectBottom.x -= PIPE_SPEED

    # getters
    def getXVal(self):
        return self.rectTop.x

    def getTopRect(self):
        return self.rectTop

    def getBottomRect(self):
        return self.rectBottom

    def getWidth(self):
        return self.WIDTH