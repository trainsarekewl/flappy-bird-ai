import pygame
import config

WINDOW_HEIGHT = config.WINDOW_HEIGHT
WINDOW_LENGTH = config.WINDOW_LENGTH
JUMP_COOLDOWN = config.JUMP_COOLDOWN
PIPE_SPEED = config.PIPE_SPEED
COLOR = config.COLOR

class Bird(pygame.sprite.Sprite):
    COLOR = (0, 0, 128)
    JUMP_HEIGHT = -5
    RADIUS = 10
    GRAVITY = 0.2

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.y_velocity = 0
        self.mask = None

    def jump(self):
        self.y_velocity = self.JUMP_HEIGHT

    def fall(self):
        self.y += self.y_velocity
        self.y_velocity += self.GRAVITY

    def draw(self, window):
        pygame.draw.circle(window, self.COLOR, (int(self.x), int(self.y)), self.RADIUS)

    def getYValue(self):
        return self.y

    def setYValue(self, yVal):
        self.y = yVal

    def setYVelocity(self, yVelo):
        self.y_velocity = yVelo

    def getRect(self):
        # For collision, return a bounding rectangle around the circle
        return pygame.Rect(
            self.x - self.RADIUS,
            self.y - self.RADIUS,
            self.RADIUS * 2,
            self.RADIUS * 2
        )
