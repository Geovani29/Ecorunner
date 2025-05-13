import pygame
from config import PLAYER_SIZE, GRAVITY, SCREEN_HEIGHT, PROJECTILE_SPEED, SCREEN_WIDTH
import os

class Player:
    def __init__(self, x, y, skin_idx=0):
        # Cargar la skin seleccionada
        self.skins = [
            pygame.image.load(os.path.join("assets", "images", f"nave{i}.png")).convert_alpha() for i in range(1, 5)
        ]
        self.image = pygame.transform.scale(self.skins[skin_idx], (60, 60))
        self.rect = self.image.get_rect(topleft=(x, y))
        self.projectiles = []
        self.health = 100
        self.speed = 5
        self.rayo_active = False
        self.rayo_timer = 0
        self.basura_count = 0

    def handle_keys(self, keys):
        # Solo movimiento vertical
        if keys[pygame.K_w] and self.rect.top > 0:
            self.rect.y -= self.speed
        if keys[pygame.K_s] and self.rect.bottom < SCREEN_HEIGHT:
            self.rect.y += self.speed

        # Limitar movimiento a los bordes verticales de la pantalla
        if self.rect.top < 0:
            self.rect.top = 0
        if self.rect.bottom > SCREEN_HEIGHT:
            self.rect.bottom = SCREEN_HEIGHT

    def shoot(self):
        projectile = pygame.Rect(self.rect.right, self.rect.centery - 5, 10, 10)
        self.projectiles.append(projectile)
        return True

    def update_projectiles(self, screen):
        for proj in self.projectiles[:]:
            proj.x += 10
            if proj.left > SCREEN_WIDTH:
                self.projectiles.remove(proj)
            pygame.draw.rect(screen, (0, 255, 0), proj)

    def draw(self, screen):
        screen.blit(self.image, self.rect.topleft)
        self.update_projectiles(screen)

    def apply_gravity(self):
        self.velocity_y += GRAVITY
        self.rect.y += self.velocity_y

        # Limitar altura inferior
        if self.rect.bottom > SCREEN_HEIGHT:
            self.rect.bottom = SCREEN_HEIGHT

        # Limitar altura superior (opcional)
        if self.rect.top < 0:
            self.rect.top = 0

    def reset(self):
        self.rect.x = 100
        self.rect.y = SCREEN_HEIGHT // 2
        self.projectiles.clear()
        self.health = 100

    def update_rayo(self):
        if self.rayo_active:
            self.rayo_timer -= 1
            if self.rayo_timer <= 0:
                self.rayo_active = False
                self.rayo_timer = 0
                self.basura_count = 0  # Reiniciar el contador de basuras
