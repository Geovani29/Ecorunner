import pygame
import random
from config import TRASH_SPEED
import os

BIRD_FRAMES = [
    pygame.transform.scale(
        pygame.image.load(os.path.join("assets", "images", f"{i}.png")),
        (70, 70)  # Cambia el tamaño a 50x50 píxeles (ajusta según lo necesites)
    )
    for i in range(1, 7)
]

EXPLOSION_FRAMES = [
    pygame.transform.scale(
        pygame.image.load(os.path.join("assets", "images", f"ex{i}.png")),
        (30, 30)  # Puedes ajustarlo luego según el tamaño del círculo
    ) for i in range(1, 10)
]

BASURA_FRAMES = [
    pygame.transform.scale(
        pygame.image.load(os.path.join("assets", "images", f"basura{i}.png")),
        (100, 100)
    )for i in range(1, 5)
]



class Trash:
    def __init__(self, x, y):
        self.image = random.choice(BASURA_FRAMES)
        self.image = pygame.transform.scale(self.image, (30, 30))  # Ajusta tamaño si quieres
        self.x = x
        self.y = y
        self.radius = 15
        self.speed = 5
        self._rect = self.image.get_rect(topleft=(self.x, self.y))

    def move(self):
        self.x -= self.speed
        self._rect.x = self.x

    def draw(self, screen):
        screen.blit(self.image, (self.x, self.y))

    def rect(self):
        return self._rect



class Obstacle:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.speed = 5  # Velocidad base que será modificada según la dificultad
        self.speed_x = -self.speed  # Velocidad horizontal inicial
        self.speed_y = 0  # Velocidad vertical inicial
        self.frame_index = 0
        self.image = BIRD_FRAMES[0]
        self.rect = self.image.get_rect(topleft=(self.x, self.y))
        self.animation_timer = 0

    def set_speed(self, new_speed):
        # Actualizar la velocidad manteniendo la dirección actual
        speed_multiplier = new_speed / self.speed
        self.speed = new_speed
        self.speed_x = -self.speed  # Siempre negativo para moverse hacia la izquierda inicialmente
        if self.speed_y != 0:  # Si ya está en movimiento diagonal
            self.speed_y = self.speed_y * speed_multiplier

    def move(self):
        self.x += self.speed_x
        self.y += self.speed_y
        self.rect.x = self.x
        self.rect.y = self.y

    def draw(self, screen):
        # Cambiar de frame cada 100 ms
        self.animation_timer += 1
        if self.animation_timer >= 5:
            self.animation_timer = 0
            self.frame_index = (self.frame_index + 1) % len(BIRD_FRAMES)
            self.image = BIRD_FRAMES[self.frame_index]

        screen.blit(self.image, (self.x, self.y))

    def rect(self):
        return self.rect

class Explosion:
    def __init__(self, x, y, size=30):
        self.frames = [
            pygame.transform.scale(frame, (size, size))
            for frame in EXPLOSION_FRAMES
        ]
        self.index = 0
        self.x = x
        self.y = y
        self.finished = False
        self.timer = 0

    def update(self):
        self.timer += 1
        if self.timer % 2 == 0:  # Cambia de frame cada 2 ticks (ajusta la velocidad aquí)
            self.index += 1
            if self.index >= len(self.frames):
                self.finished = True

    def draw(self, screen):
        if not self.finished:
            screen.blit(self.frames[self.index], (self.x, self.y))
