import pygame
import config

WINDOW_HEIGHT = config.WINDOW_HEIGHT
WINDOW_LENGTH = config.WINDOW_LENGTH
JUMP_COOLDOWN = config.JUMP_COOLDOWN
PIPE_SPEED = config.PIPE_SPEED
COLOR = config.COLOR
BASE_DIR = config.BASE_DIR

class Bird(pygame.sprite.Sprite):

    # class variables
    COLOR = (0, 255, 12)
    JUMP_HEIGHT = -5
    SIZE = 15
    GRAVITY = 0.2

    def __init__(self, x, y):
        self.rect = pygame.Rect(x, y, self.SIZE, self.SIZE)
        self.y_velocity = 0
        self.image = pygame.image.load(BASE_DIR / "assets/flappybird.png")
        self.mask = None

    def jump(self):
        self.y_velocity = self.JUMP_HEIGHT

    def fall(self):
        self.rect.y += self.y_velocity
        self.y_velocity += self.GRAVITY

    def draw(self, window):
        window.blit(self.image, (self.rect.x - 10, self.rect.y - 5))
        # pygame.draw.rect(window, self.COLOR, self.rect)

    def getYValue(self):
        return self.rect.y

    def setYValue(self, yVal):
        self.rect.y = yVal

    def setYVelocity(self, yVelo):
        self.y_velocity = yVelo

    def getRect(self):
        return self.rect