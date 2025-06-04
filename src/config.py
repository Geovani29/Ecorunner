import pygame

# Obtener dimensiones de la pantalla
pygame.init()
info = pygame.display.Info()
SCREEN_WIDTH = info.current_w
SCREEN_HEIGHT = info.current_h

# Configuración general
FPS = 60

PLAYER_SIZE = 50
GRAVITY = 0.5
PROJECTILE_SPEED = 10
TRASH_SPEED = 5
