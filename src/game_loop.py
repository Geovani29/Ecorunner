import pygame
import random
import os
from config import SCREEN_WIDTH, SCREEN_HEIGHT, FPS
from player import Player
from items import Trash, Obstacle, Explosion
from rl.agent_runner import QLearningAgent
import numpy as np
import math
import sys

pygame.font.init()
pygame.mixer.init()

# Cargar la fuente pixelada Press Start 2P en diferentes tamaños (ajustados)
FONT_BIG = pygame.font.Font("assets/fonts/PressStart2P-Regular.ttf", 48)
FONT_TITLE = pygame.font.Font("assets/fonts/PressStart2P-Regular.ttf", 22)
FONT_UI = pygame.font.Font("assets/fonts/PressStart2P-Regular.ttf", 20)
FONT_BUTTON = pygame.font.Font("assets/fonts/PressStart2P-Regular.ttf", 12)
FONT_SMALL = pygame.font.Font("assets/fonts/PressStart2P-Regular.ttf", 12)

# Reemplazar FONT = pygame.font.SysFont("Arial", 30)
FONT = FONT_UI

laser_sound = pygame.mixer.Sound("assets/sounds/laser.mp3")
explosion_sound = pygame.mixer.Sound("assets/sounds/explo.wav")
laser2_sound = pygame.mixer.Sound("assets/sounds/laser2.mp3")
menu_background = pygame.image.load("assets/images/fondo.png")
menu_background = pygame.transform.scale(menu_background, (SCREEN_WIDTH, SCREEN_HEIGHT))
menu_background2 = pygame.image.load("assets/images/fondo2.png")
menu_background2 = pygame.transform.scale(menu_background2, (SCREEN_WIDTH, SCREEN_HEIGHT))

# Cargar imágenes de botones
button_imgs = [
	pygame.image.load("assets/images/boton1.png"),
	pygame.image.load("assets/images/boton2.png"),
	pygame.image.load("assets/images/boton3.png"),
	pygame.image.load("assets/images/boton4.png")
]
button_imgs = [pygame.transform.scale(img, (320, 56)) for img in button_imgs]

# Cargar imágenes de skins de nave
nave_skins = [
    pygame.image.load(f"assets/images/nave{i}.png") for i in range(1, 5)
]
nave_skins = [pygame.transform.scale(img, (64, 64)) for img in nave_skins]

selected_skin_idx = 0  # Índice de skin seleccionada por defecto

trofeo_img = pygame.image.load("assets/images/trofeo.png")
trofeo_img = pygame.transform.scale(trofeo_img, (40, 40))

def draw_text_center(screen, text, font, color, y_offset=0):
	text_surface = font.render(text, True, color)
	rect = text_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + y_offset))
	screen.blit(text_surface, rect)

def draw_button(screen, rect, text, is_hovered=False):
	base_color = (0, 102, 204)
	hover_color = (0, 82, 184)
	text_color = (255, 255, 255)
	shadow_rect = rect.copy()
	shadow_rect.x += 2
	shadow_rect.y += 2
	pygame.draw.rect(screen, (0, 0, 0, 128), shadow_rect, border_radius=15)
	current_color = hover_color if is_hovered else base_color
	pygame.draw.rect(screen, current_color, rect, border_radius=15)
	highlight_rect = rect.copy()
	highlight_rect.width = 3
	highlight_rect.height = 3
	pygame.draw.rect(screen, (255, 255, 255, 180), highlight_rect, border_radius=15)
	button_text = FONT_BUTTON.render(text, True, text_color)
	text_rect = button_text.get_rect(center=rect.center)
	shadow_text = FONT_BUTTON.render(text, True, (0, 0, 0, 128))
	shadow_rect = text_rect.copy()
	shadow_rect.x += 1
	shadow_rect.y += 1
	screen.blit(shadow_text, shadow_rect)
	screen.blit(button_text, text_rect)

def draw_gradient_title(screen, text, y_offset=0):
	# Título con gradiente de colores
	title_font = FONT_TITLE
	main_text = title_font.render(text, True, (255, 255, 255))
	main_rect = main_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 - 120 + y_offset))
	# Crear gradiente horizontal
	grad = np.linspace(0, 1, main_rect.width)
	for i, g in enumerate(grad):
		color = (
			int(255 * (1-g) + 0 * g),   # Rojo a azul
			int(200 * g),                # Verde
			int(255 * g)                 # Azul
		)
		pygame.draw.line(screen, color, (main_rect.left + i, main_rect.top), (main_rect.left + i, main_rect.bottom))
	# Sombra negra
	for offset_x, offset_y in [(1,1), (1,-1), (-1,1), (-1,-1), (1,0), (-1,0), (0,1), (0,-1)]:
		shadow_text = title_font.render(text, True, (0, 0, 0))
		shadow_rect = shadow_text.get_rect(center=(SCREEN_WIDTH//2 + offset_x, SCREEN_HEIGHT//2 - 120 + offset_y + y_offset))
		screen.blit(shadow_text, shadow_rect)
	# Texto principal (blanco, sobre el gradiente)
	screen.blit(main_text, main_rect)

def draw_colored_button(screen, rect, text, color, is_hovered=False):
	# Botón con color personalizado y efecto hover
	base_color = color
	hover_color = tuple(min(255, int(c*1.15)) for c in color)
	text_color = (255, 255, 255)
	shadow_rect = rect.copy()
	shadow_rect.x += 2
	shadow_rect.y += 2
	pygame.draw.rect(screen, (0, 0, 0, 80), shadow_rect, border_radius=12)
	current_color = hover_color if is_hovered else base_color
	pygame.draw.rect(screen, current_color, rect, border_radius=12)
	pygame.draw.rect(screen, (255,255,255,80), rect, width=2, border_radius=12)
	button_text = FONT_BUTTON.render(text, True, text_color)
	text_rect = button_text.get_rect(center=rect.center)
	shadow_text = FONT_BUTTON.render(text, True, (0, 0, 0, 128))
	shadow_rect = text_rect.copy()
	shadow_rect.x += 1
	shadow_rect.y += 1
	screen.blit(shadow_text, shadow_rect)
	screen.blit(button_text, text_rect)

def draw_soft_button(screen, rect, text, is_hovered=False):
	# Botón blanco semitransparente, texto negro, borde suave, sombra y hover azul
	base_color = (255, 255, 255, 180)
	hover_color = (180, 220, 255, 220)
	text_color = (30, 30, 30)
	border_color = (0, 102, 204)
	shadow_rect = rect.copy()
	shadow_rect.x += 2
	shadow_rect.y += 2
	pygame.draw.rect(screen, (0, 0, 0, 60), shadow_rect, border_radius=12)
	current_color = hover_color if is_hovered else base_color
	# Superficie para alpha
	button_surface = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
	button_surface.fill(current_color)
	screen.blit(button_surface, rect.topleft)
	pygame.draw.rect(screen, border_color, rect, width=2, border_radius=12)
	button_text = FONT_BUTTON.render(text, True, text_color)
	text_rect = button_text.get_rect(center=rect.center)
	shadow_text = FONT_BUTTON.render(text, True, (255,255,255,80))
	shadow_rect = text_rect.copy()
	shadow_rect.x += 1
	shadow_rect.y += 1
	screen.blit(shadow_text, shadow_rect)
	screen.blit(button_text, text_rect)

def draw_help_messages(screen):
	help_font = FONT_SMALL
	help_color = (255, 255, 255)
	help_shadow = (0, 0, 0)
	
	# Lista de mensajes de ayuda con sus posiciones
	help_messages = [
		("↑ W: Mover arriba", SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 200),
		("↓ S: Mover abajo", SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 230),
		("Click Izquierdo: Disparar", SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 260),
		("Evita la basura y los pájaros", SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 290)
	]
	
	# Dibujar cada mensaje con sombra
	for text, x, y in help_messages:
		shadow = help_font.render(text, True, help_shadow)
		text_surface = help_font.render(text, True, help_color)
		shadow_rect = shadow.get_rect(center=(x+1, y+1))
		text_rect = text_surface.get_rect(center=(x, y))
		screen.blit(shadow, shadow_rect)
		screen.blit(text_surface, text_rect)

def draw_contextual_help(screen, state):
	help_font = FONT_SMALL
	help_color = (255, 255, 255)
	help_shadow = (0, 0, 0)
	
	# Posiciones fijas para los mensajes en la parte superior
	y_position = 50  # Posición fija en la parte superior
	
	# Mensajes de ayuda contextuales
	if not state.get("moved_up", False):
		text = "↑ W: Mover arriba"
		shadow = help_font.render(text, True, help_shadow)
		text_surface = help_font.render(text, True, help_color)
		shadow_rect = shadow.get_rect(center=(SCREEN_WIDTH//2+1, y_position+1))
		text_rect = text_surface.get_rect(center=(SCREEN_WIDTH//2, y_position))
		screen.blit(shadow, shadow_rect)
		screen.blit(text_surface, text_rect)
	
	if not state.get("moved_down", False):
		text = "↓ S: Mover abajo"
		shadow = help_font.render(text, True, help_shadow)
		text_surface = help_font.render(text, True, help_color)
		shadow_rect = shadow.get_rect(center=(SCREEN_WIDTH//2+1, y_position+30+1))
		text_rect = text_surface.get_rect(center=(SCREEN_WIDTH//2, y_position+30))
		screen.blit(shadow, shadow_rect)
		screen.blit(text_surface, text_rect)
	
	if not state.get("shot", False):
		text = "Click Izquierdo: Disparar"
		shadow = help_font.render(text, True, help_shadow)
		text_surface = help_font.render(text, True, help_color)
		shadow_rect = shadow.get_rect(center=(SCREEN_WIDTH//2+1, y_position+60+1))
		text_rect = text_surface.get_rect(center=(SCREEN_WIDTH//2, y_position+60))
		screen.blit(shadow, shadow_rect)
		screen.blit(text_surface, text_rect)

def draw_pixel_button(screen, rect, color, text, font, is_hovered=False, shadow_color=None):
	border_color = (0, 0, 0)
	if not shadow_color:
		shadow_color = tuple(max(0, c-40) for c in color)
	scale = 4
	# Cuerpo principal
	pygame.draw.rect(screen, color, rect)
	# Borde negro escalonado (pixel art)
	# Superior
	for i in range(rect.width//scale):
		pygame.draw.rect(screen, border_color, (rect.x + i*scale, rect.y, scale, scale))
	# Inferior
	for i in range(rect.width//scale):
		pygame.draw.rect(screen, border_color, (rect.x + i*scale, rect.bottom-scale, scale, scale))
	# Izquierda
	for i in range(rect.height//scale):
		pygame.draw.rect(screen, border_color, (rect.x, rect.y + i*scale, scale, scale))
	# Derecha
	for i in range(rect.height//scale):
		pygame.draw.rect(screen, border_color, (rect.right-scale, rect.y + i*scale, scale, scale))
	# Esquinas escalonadas (patrón auténtico)
	# Superior izquierda
	pygame.draw.rect(screen, border_color, (rect.x, rect.y, scale, scale))
	pygame.draw.rect(screen, border_color, (rect.x+scale, rect.y, scale, scale))
	pygame.draw.rect(screen, border_color, (rect.x, rect.y+scale, scale, scale))
	pygame.draw.rect(screen, border_color, (rect.x+2*scale, rect.y, scale, scale))
	pygame.draw.rect(screen, border_color, (rect.x, rect.y+2*scale, scale, scale))
	# Superior derecha
	pygame.draw.rect(screen, border_color, (rect.right-scale, rect.y, scale, scale))
	pygame.draw.rect(screen, border_color, (rect.right-2*scale, rect.y, scale, scale))
	pygame.draw.rect(screen, border_color, (rect.right-scale, rect.y+scale, scale, scale))
	pygame.draw.rect(screen, border_color, (rect.right-3*scale, rect.y, scale, scale))
	pygame.draw.rect(screen, border_color, (rect.right-scale, rect.y+2*scale, scale, scale))
	# Inferior izquierda
	pygame.draw.rect(screen, border_color, (rect.x, rect.bottom-scale, scale, scale))
	pygame.draw.rect(screen, border_color, (rect.x+scale, rect.bottom-scale, scale, scale))
	pygame.draw.rect(screen, border_color, (rect.x, rect.bottom-2*scale, scale, scale))
	pygame.draw.rect(screen, border_color, (rect.x+2*scale, rect.bottom-scale, scale, scale))
	pygame.draw.rect(screen, border_color, (rect.x, rect.bottom-3*scale, scale, scale))
	# Inferior derecha
	pygame.draw.rect(screen, border_color, (rect.right-scale, rect.bottom-scale, scale, scale))
	pygame.draw.rect(screen, border_color, (rect.right-2*scale, rect.bottom-scale, scale, scale))
	pygame.draw.rect(screen, border_color, (rect.right-scale, rect.bottom-2*scale, scale, scale))
	pygame.draw.rect(screen, border_color, (rect.right-3*scale, rect.bottom-scale, scale, scale))
	pygame.draw.rect(screen, border_color, (rect.right-scale, rect.bottom-3*scale, scale, scale))
	# Sombra escalonada en esquinas
	# Superior izquierda
	pygame.draw.rect(screen, shadow_color, (rect.x, rect.y, scale, scale))
	pygame.draw.rect(screen, shadow_color, (rect.x+scale, rect.y, scale, scale))
	pygame.draw.rect(screen, shadow_color, (rect.x, rect.y+scale, scale, scale))
	# Superior derecha
	pygame.draw.rect(screen, shadow_color, (rect.right-scale, rect.y, scale, scale))
	pygame.draw.rect(screen, shadow_color, (rect.right-2*scale, rect.y, scale, scale))
	pygame.draw.rect(screen, shadow_color, (rect.right-scale, rect.y+scale, scale, scale))
	# Inferior izquierda
	pygame.draw.rect(screen, shadow_color, (rect.x, rect.bottom-scale, scale, scale))
	pygame.draw.rect(screen, shadow_color, (rect.x+scale, rect.bottom-scale, scale, scale))
	pygame.draw.rect(screen, shadow_color, (rect.x, rect.bottom-2*scale, scale, scale))
	# Inferior derecha
	pygame.draw.rect(screen, shadow_color, (rect.right-scale, rect.bottom-scale, scale, scale))
	pygame.draw.rect(screen, shadow_color, (rect.right-2*scale, rect.bottom-scale, scale, scale))
	pygame.draw.rect(screen, shadow_color, (rect.right-scale, rect.bottom-2*scale, scale, scale))
	# Texto centrado
	text_surface = font.render(text, True, (0,0,0))
	text_rect = text_surface.get_rect(center=rect.center)
	screen.blit(text_surface, text_rect)

def draw_image_button(screen, rect, img, text, font, is_hovered=False):
	# Dibuja el botón usando una imagen de fondo y texto centrado, adaptando el tamaño de la fuente
	screen.blit(img, rect.topleft)
	# Ajustar tamaño de fuente dinámicamente
	max_font_size = min(24, rect.height - 12)  # Limitar a 24 px máximo
	min_font_size = 16
	font_size = max_font_size
	button_font = None
	text_surface = None
	while font_size >= min_font_size:
		button_font = pygame.font.Font("assets/fonts/PressStart2P-Regular.ttf", font_size)
		text_surface = button_font.render(text, True, (0,0,0))
		if text_surface.get_width() <= rect.width - 24:
			break
		font_size -= 2
	text_rect = text_surface.get_rect(center=rect.center)
	screen.blit(text_surface, text_rect)

def draw_help_box(screen, instructions, font):
	# Dibuja un recuadro centrado estilo pixel art con instrucciones
	box_width, box_height = 700, 580
	box_x = SCREEN_WIDTH//2 - box_width//2
	box_y = SCREEN_HEIGHT//2 - box_height//2
	scale = 5
	padding_x = 44
	padding_y = 38
	line_spacing = 44
	max_text_width = box_width - 2*padding_x
	# Fondo
	pygame.draw.rect(screen, (30, 30, 60), (box_x, box_y, box_width, box_height), border_radius=18)
	# Borde negro escalonado
	for i in range(box_width//scale):
		pygame.draw.rect(screen, (0,0,0), (box_x + i*scale, box_y, scale, scale),)
		pygame.draw.rect(screen, (0,0,0), (box_x + i*scale, box_y+box_height-scale, scale, scale))
	for i in range(box_height//scale):
		pygame.draw.rect(screen, (0,0,0), (box_x, box_y + i*scale, scale, scale))
		pygame.draw.rect(screen, (0,0,0), (box_x+box_width-scale, box_y + i*scale, scale, scale))
	# Esquinas escalonadas
	for dx, dy in [(0,0),(scale,0),(0,scale),(box_width-scale,0),(box_width-2*scale,0),(box_width-scale,scale),
				   (0,box_height-scale),(scale,box_height-scale),(0,box_height-2*scale),
				   (box_width-scale,box_height-scale),(box_width-2*scale,box_height-scale),(box_width-scale,box_height-2*scale)]:
		pygame.draw.rect(screen, (0,0,0), (box_x+dx, box_y+dy, scale, scale))
	# Título
	title = font.render("INSTRUCCIONES", True, (255,255,0))
	screen.blit(title, (box_x+box_width//2-title.get_width()//2, box_y+padding_y))
	# Ajustar fuente y hacer word wrap
	help_font_size = 14
	help_font = pygame.font.Font("assets/fonts/PressStart2P-Regular.ttf", help_font_size)
	wrapped_lines = []
	for line in instructions:
		words = line.split(' ')
		current_line = ''
		for word in words:
			test_line = current_line + (' ' if current_line else '') + word
			test_surface = help_font.render(test_line, True, (255,255,255))
			if test_surface.get_width() > max_text_width:
				if current_line:
					wrapped_lines.append(current_line)
				current_line = word
			else:
				current_line = test_line
		if current_line:
			wrapped_lines.append(current_line)
	# Si aún así alguna línea es muy larga, reducir fuente
	while any(help_font.render(l, True, (255,255,255)).get_width() > max_text_width for l in wrapped_lines) and help_font_size > 12:
		help_font_size -= 1
		help_font = pygame.font.Font("assets/fonts/PressStart2P-Regular.ttf", help_font_size)
	# Dibujar líneas
	y = box_y+padding_y+48
	for line in wrapped_lines:
		text = help_font.render(line, True, (255,255,255))
		screen.blit(text, (box_x+padding_x, y))
		y += line_spacing
	# Botón cerrar
	close_rect = pygame.Rect(box_x+box_width-160, box_y+box_height-70, 140, 44)
	pygame.draw.rect(screen, (255,180,40), close_rect, border_radius=12)
	pygame.draw.rect(screen, (0,0,0), close_rect, 4, border_radius=12)
	close_font = pygame.font.Font("assets/fonts/PressStart2P-Regular.ttf", 20)
	close_text = close_font.render("CERRAR", True, (0,0,0))
	screen.blit(close_text, (close_rect.centerx-close_text.get_width()//2, close_rect.centery-close_text.get_height()//2))
	return close_rect

def draw_skin_selector(screen, current_idx):
    # Ventana centrada estilo pixel art
    box_width, box_height = 480, 220
    box_x = SCREEN_WIDTH//2 - box_width//2
    box_y = SCREEN_HEIGHT//2 - box_height//2
    scale = 5
    # Fondo y borde
    pygame.draw.rect(screen, (30, 30, 60), (box_x, box_y, box_width, box_height), border_radius=18)
    pygame.draw.rect(screen, (0,0,0), (box_x, box_y, box_width, box_height), scale, border_radius=18)
    # Título
    font = pygame.font.Font("assets/fonts/PressStart2P-Regular.ttf", 20)
    title = font.render("ELIGE TU NAVE", True, (255,255,0))
    screen.blit(title, (box_x+box_width//2-title.get_width()//2, box_y+18))
    # Mostrar las 4 naves
    skin_rects = []
    for i, img in enumerate(nave_skins):
        x = box_x + 40 + i*110
        y = box_y + 70
        rect = pygame.Rect(x, y, 64, 64)
        skin_rects.append(rect)
        screen.blit(img, rect.topleft)
        # Resaltar la seleccionada
        if i == current_idx:
            pygame.draw.rect(screen, (255,255,0), rect.inflate(10,10), 4, border_radius=10)
        else:
            pygame.draw.rect(screen, (255,255,255), rect, 2, border_radius=10)
    # Botón cerrar
    close_rect = pygame.Rect(box_x+box_width-160, box_y+box_height-50, 130, 36)
    pygame.draw.rect(screen, (255,180,40), close_rect, border_radius=12)
    pygame.draw.rect(screen, (0,0,0), close_rect, 4, border_radius=12)
    close_font = pygame.font.Font("assets/fonts/PressStart2P-Regular.ttf", 18)
    close_text = close_font.render("CERRAR", True, (0,0,0))
    screen.blit(close_text, (close_rect.centerx-close_text.get_width()//2, close_rect.centery-close_text.get_height()//2))
    return skin_rects, close_rect

def draw_pixel_trophy(screen, x, y, scale=2):
    # Trofeo pixel art simple
    color = (255, 215, 0)
    pygame.draw.rect(screen, color, (x+2*scale, y+6*scale, 4*scale, 6*scale)) # base
    pygame.draw.rect(screen, color, (x, y+2*scale, 8*scale, 6*scale)) # copa
    pygame.draw.rect(screen, color, (x-2*scale, y+4*scale, 2*scale, 2*scale)) # asa izq
    pygame.draw.rect(screen, color, (x+8*scale, y+4*scale, 2*scale, 2*scale)) # asa der
    pygame.draw.rect(screen, (255,255,255), (x+2*scale, y+3*scale, scale, scale)) # brillo

def mostrar_ranking_modal(screen, fondo_surface=None):
    import csv
    ranking = {"Fácil":[], "Normal":[], "Difícil":[]}
    try:
        with open("data/scores.csv", "r", encoding='utf-8') as f:
            for row in csv.reader(f):
                # Ignorar líneas vacías o mal formateadas
                if not row or len(row) < 3:
                    continue
                try:
                    nombre, dificultad, puntaje = row
                    # Normalizar el nombre de la dificultad
                    dificultad = dificultad.strip()  # Eliminar espacios en blanco
                    puntaje = int(puntaje)
                    if dificultad in ranking:
                        ranking[dificultad].append((nombre, puntaje))
                except (ValueError, IndexError):
                    continue
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"Error al leer el archivo de puntuaciones: {e}")

    # Ordenar cada lista de puntuaciones
    for key in ranking:
        ranking[key].sort(key=lambda x: x[1], reverse=True)

    clock = pygame.time.Clock()
    salir = False
    while not salir:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                if cerrar_btn.collidepoint(mx, my):
                    salir = True
        # Fondo: si hay fondo_surface, usarlo, si no, negro
        if fondo_surface:
            screen.blit(fondo_surface, (0, 0))
        else:
            screen.fill((0, 0, 0))
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        overlay.set_alpha(180)
        overlay.fill((0, 0, 0))
        screen.blit(overlay, (0, 0))
        win_w, win_h = 700, 400
        win_x = (SCREEN_WIDTH - win_w) // 2
        win_y = (SCREEN_HEIGHT - win_h) // 2
        pygame.draw.rect(screen, (40, 40, 60), (win_x, win_y, win_w, win_h), border_radius=16)
        pygame.draw.rect(screen, (255,255,255), (win_x, win_y, win_w, win_h), 3, border_radius=16)
        titulo = FONT_TITLE.render("Ranking", True, (255,255,0))
        screen.blit(titulo, (win_x + (win_w-titulo.get_width())//2, win_y+20))
        col_w = win_w // 3
        col_titles = ["Fácil", "Normal", "Difícil"]
        for i, dificultad in enumerate(col_titles):
            col_x = win_x + i*col_w
            subt = FONT_UI.render(dificultad, True, (0,255,255))
            screen.blit(subt, (col_x + (col_w-subt.get_width())//2, win_y+60))
            for j, (nombre, puntaje) in enumerate(ranking[dificultad][:5], 1):
                linea = FONT_SMALL.render(f"{j}. {nombre}: {puntaje}", True, (255,255,255))
                screen.blit(linea, (col_x + 20, win_y+100 + (j-1)*28))
            if i < 2:
                pygame.draw.line(screen, (180,180,180), (col_x+col_w, win_y+50), (col_x+col_w, win_y+win_h-80), 3)
        cerrar_btn = pygame.Rect(win_x+win_w//2-60, win_y+win_h-60, 120, 36)
        pygame.draw.rect(screen, (180,0,0), cerrar_btn, border_radius=8)
        pygame.draw.rect(screen, (0,0,0), cerrar_btn, 2, border_radius=8)
        cerrar_txt = FONT_BUTTON.render("Cerrar", True, (255,255,255))
        screen.blit(cerrar_txt, (cerrar_btn.centerx-cerrar_txt.get_width()//2, cerrar_btn.centery-cerrar_txt.get_height()//2))
        pygame.display.flip()
        clock.tick(60)

def show_difficulty_menu(screen):
    global selected_skin_idx
    selected_mode = None
    bg = pygame.transform.smoothscale(menu_background, (SCREEN_WIDTH, SCREEN_HEIGHT))
    ia_submenu = False
    show_help = False
    show_skin_selector = False
    show_ranking = False
    help_instructions = [
        "- Usa W/S o ↑/↓ para mover la nave",
        "- Dispara con click izquierdo",
        "- Recolecta basura para sumar puntos",
        "- Si pierdes basura, el planeta se oscurece",
        "- Escudo: 5 basuras seguidas",
        "- Rayo: 10 basuras seguidas, click derecho",
        "- Evita los pájaros y obstáculos",
        "- El juego termina si tu salud llega a 0"
    ]
    while not selected_mode:
        screen.blit(bg, (0, 0))
        draw_gradient_title(screen, "Selecciona la dificultad o modo de IA", -40)
        # Botón de ranking arriba a la izquierda
        ranking_btn = pygame.Rect(20, 20, 140, 48)
        pygame.draw.rect(screen, (255, 220, 40), ranking_btn, border_radius=12)
        pygame.draw.rect(screen, (0,0,0), ranking_btn, 4, border_radius=12)
        screen.blit(trofeo_img, (ranking_btn.x+8, ranking_btn.y+4))
        ranking_txt = FONT_BUTTON.render("Ranking", True, (0,0,0))
        screen.blit(ranking_txt, (ranking_btn.x+48, ranking_btn.y+ranking_btn.height//2-ranking_txt.get_height()//2))
        # Botón de ayuda en la esquina superior derecha
        help_btn_rect = pygame.Rect(SCREEN_WIDTH-70, 20, 48, 48)
        pygame.draw.rect(screen, (255, 220, 40), help_btn_rect, border_radius=12)
        pygame.draw.rect(screen, (0,0,0), help_btn_rect, 4, border_radius=12)
        q_font = pygame.font.Font("assets/fonts/PressStart2P-Regular.ttf", 32)
        q_text = q_font.render("?", True, (0,0,0))
        screen.blit(q_text, (help_btn_rect.centerx-q_text.get_width()//2, help_btn_rect.centery-q_text.get_height()//2))
        # Botón elegir skin
        skin_btn_rect = pygame.Rect(SCREEN_WIDTH-210, 20, 120, 48)
        pygame.draw.rect(screen, (70,200,255), skin_btn_rect, border_radius=12)
        pygame.draw.rect(screen, (0,0,0), skin_btn_rect, 4, border_radius=12)
        skin_font = pygame.font.Font("assets/fonts/PressStart2P-Regular.ttf", 16)
        skin_text = skin_font.render("Skins", True, (0,0,0))
        screen.blit(skin_text, (skin_btn_rect.centerx-skin_text.get_width()//2, skin_btn_rect.centery-skin_text.get_height()//2))
        if show_help:
            close_rect = draw_help_box(screen, help_instructions, FONT_BUTTON)
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = pygame.mouse.get_pos()
                    if close_rect.collidepoint(mx, my):
                        show_help = False
            continue
        if show_skin_selector:
            skin_rects, close_rect = draw_skin_selector(screen, selected_skin_idx)
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = pygame.mouse.get_pos()
                    for idx, rect in enumerate(skin_rects):
                        if rect.collidepoint(mx, my):
                            selected_skin_idx = idx
                    if close_rect.collidepoint(mx, my):
                        show_skin_selector = False
            continue
        if show_ranking:
            # Renderizar la pantalla de dificultad como fondo para el ranking
            fondo_surface = screen.copy()
            mostrar_ranking_modal(screen, fondo_surface)
            show_ranking = False
            continue
        if not ia_submenu:
            button_data = [
                (button_imgs[0], "Fácil"),
                (button_imgs[1], "Normal"),
                (button_imgs[2], "Difícil"),
                (button_imgs[3], "IA")
            ]
        else:
            button_data = [
                (button_imgs[3], "IA - Manual"),
                (button_imgs[3], "IA - Automática")
            ]
        buttons = {}
        start_y = SCREEN_HEIGHT // 2 - (140 if not ia_submenu else 40)
        button_width = 320
        button_height = 56
        button_gap = 28
        mouse_pos = pygame.mouse.get_pos()
        for i, (img, label) in enumerate(button_data):
            rect = pygame.Rect(SCREEN_WIDTH//2 - button_width//2, start_y + i * (button_height + button_gap), button_width, button_height)
            buttons[label] = rect
            is_hovered = rect.collidepoint(mouse_pos)
            draw_image_button(screen, rect, img, label, FONT_BIG, is_hovered)
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                if ranking_btn.collidepoint(mx, my):
                    show_ranking = True
                    break
                if help_btn_rect.collidepoint(mx, my):
                    show_help = True
                    break
                if skin_btn_rect.collidepoint(mx, my):
                    show_skin_selector = True
                    break
                for label, rect in buttons.items():
                    if rect.collidepoint(mx, my):
                        if label == "IA" and not ia_submenu:
                            ia_submenu = True
                            break
                        elif label == "IA - Manual":
                            return "IA", "Manual", selected_skin_idx
                        elif label == "IA - Automática":
                            return "IA", "Automatica", selected_skin_idx
                        else:
                            return label, None, selected_skin_idx

def draw_menu_button(screen, mouse_pos):
	menu_btn = pygame.Rect(SCREEN_WIDTH - 230, 10, 210, 36)
	is_hovered = menu_btn.collidepoint(mouse_pos)
	
	# Color base y color hover
	base_color = (0, 102, 204)  # Azul
	hover_color = (0, 82, 184)  # Azul más oscuro
	text_color = (255, 255, 255)  # Blanco
	
	# Dibujar sombra con un radio mayor
	shadow_rect = menu_btn.copy()
	shadow_rect.x += 2
	shadow_rect.y += 2
	pygame.draw.rect(screen, (0, 0, 0, 128), shadow_rect, border_radius=15)
	
	# Dibujar el botón con efecto hover
	current_color = hover_color if is_hovered else base_color
	pygame.draw.rect(screen, current_color, menu_btn, border_radius=15)
	
	# Añadir borde brillante con un radio mayor
	highlight_rect = menu_btn.copy()
	highlight_rect.width = 3
	highlight_rect.height = 3
	pygame.draw.rect(screen, (255, 255, 255, 180), highlight_rect, border_radius=15)
	
	# Texto con efecto de sombra
	menu_text = FONT_BUTTON.render("Volver al menú", True, text_color)
	text_rect = menu_text.get_rect(center=menu_btn.center)
	
	# Sombra del texto más suave
	shadow_text = FONT_BUTTON.render("Volver al menú", True, (0, 0, 0, 128))
	shadow_rect = text_rect.copy()
	shadow_rect.x += 1
	shadow_rect.y += 1
	screen.blit(shadow_text, shadow_rect)
	
	# Texto principal
	screen.blit(menu_text, text_rect)
	
	return menu_btn

def draw_pixel_heart(surface, x, y, scale=2):
	# Dibuja un corazón pixelado (8x8) solo en rojo, sin bordes blancos
	heart_pixels = [
		"01100110",
		"11111111",
		"11111111",
		"11111111",
		"01111110",
		"00111100",
		"00011000",
		"00000000"
	]
	for row, line in enumerate(heart_pixels):
		for col, char in enumerate(line):
			if char == "1":
				pygame.draw.rect(surface, (220, 0, 0), (x + col*scale, y + row*scale, scale, scale))

def draw_game_ui(screen, score, health, consecutive_trash, cleaner_ray_ready):
	UI_FONT = FONT_UI
	# Barra de vida estilo pixel art
	bar_x = 50  # Separar más la barra del corazón
	bar_y = 14
	bar_width = 180
	bar_height = 14
	border = 2
	# Fondo gris
	pygame.draw.rect(screen, (120,120,120), (bar_x, bar_y, bar_width, bar_height))
	# Relleno verde según la vida
	fill_width = int((health/100) * (bar_width-4))
	pygame.draw.rect(screen, (170, 220, 40), (bar_x+2, bar_y+2, fill_width, bar_height-4))
	# Borde blanco
	pygame.draw.rect(screen, (255,255,255), (bar_x, bar_y, bar_width, bar_height), border)
	# Punta blanca al final
	pygame.draw.rect(screen, (255,255,255), (bar_x+bar_width-4, bar_y+2, 4, bar_height-4))
	# Corazón pixelado (centrado verticalmente respecto a la barra)
	draw_pixel_heart(screen, 10, bar_y-2, scale=3)
	# Puntuación
	score_text = UI_FONT.render(f"Puntos: {score}", True, (255, 255, 255))
	score_shadow = UI_FONT.render(f"Puntos: {score}", True, (0, 0, 0))
	screen.blit(score_shadow, (11, 35))
	screen.blit(score_text, (10, 34))
	# Estado del rayo
	ray_status = "¡RAYO LISTO!" if cleaner_ray_ready else f"Rayo: {consecutive_trash}/10"
	ray_color = (0, 255, 255) if cleaner_ray_ready else (255, 255, 255)
	ray_text = UI_FONT.render(ray_status, True, ray_color)
	ray_shadow = UI_FONT.render(ray_status, True, (0, 0, 0))
	screen.blit(ray_shadow, (11, 60))
	screen.blit(ray_text, (10, 59))

def draw_game_over(screen, score, auto_mode):
	# Crear un efecto de fondo para el Game Over
	overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
	overlay.fill((0, 0, 0))
	overlay.set_alpha(180)
	screen.blit(overlay, (0, 0))
	
	# Dibujar el texto "GAME OVER" con estilo futurista
	game_over_font = pygame.font.SysFont("Arial", 80, bold=True)
	
	# Efecto de brillo para GAME OVER
	for i in range(3):
		glow_text = game_over_font.render("GAME OVER", True, (0, 255, 255, 100 - i*30))
		glow_rect = glow_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 - 50))
		screen.blit(glow_text, glow_rect)
	
	# Texto principal de GAME OVER
	main_text = game_over_font.render("GAME OVER", True, (255, 255, 255))
	main_rect = main_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 - 50))
	screen.blit(main_text, main_rect)
	
	# Mostrar puntuación final
	score_font = pygame.font.SysFont("Arial", 40, bold=True)
	score_text = score_font.render(f"Puntuación Final: {score}", True, (255, 255, 255))
	score_rect = score_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 20))
	screen.blit(score_text, score_rect)
	
	if not auto_mode:
		# Botón de reiniciar con el mismo estilo que los demás
		btn_rect = pygame.Rect(SCREEN_WIDTH//2 - 100, SCREEN_HEIGHT//2 + 80, 200, 40)
		mouse_pos = pygame.mouse.get_pos()
		is_hovered = btn_rect.collidepoint(mouse_pos)
		draw_button(screen, btn_rect, "REINICIAR", is_hovered)
		return btn_rect
	return None

def draw_game_over_glitch(screen, score):
	# Fondo oscuro con estrellas
	screen.fill((15, 15, 20))
	for _ in range(120):
		x = random.randint(0, SCREEN_WIDTH)
		y = random.randint(0, SCREEN_HEIGHT)
		color = random.choice([(255,255,255), (180,180,180), (100,100,100)])
		screen.fill(color, ((x, y), (2, 2)))

	# Efecto glitch para GAME OVER
	text = "GAME OVER"
	base_x = SCREEN_WIDTH // 2
	base_y = SCREEN_HEIGHT // 2 - 80
	for dx, color in [(-4, (0,255,255)), (4, (255,0,0))]:
		glitch = FONT_BIG.render(text, True, color)
		rect = glitch.get_rect(center=(base_x+dx, base_y))
		screen.blit(glitch, rect)
	main = FONT_BIG.render(text, True, (255,255,255))
	main_rect = main.get_rect(center=(base_x, base_y))
	screen.blit(main, main_rect)

	# PLAY AGAIN?
	play_again = FONT_SMALL.render("PLAY AGAIN?", True, (255,255,255))
	play_rect = play_again.get_rect(center=(base_x, base_y+70))
	pygame.draw.rect(screen, (0,0,0), play_rect.inflate(30,10))
	screen.blit(play_again, play_rect)

	# Botones YES/NO
	yes_rect = pygame.Rect(base_x-70, base_y+120, 60, 36)
	no_rect = pygame.Rect(base_x+10, base_y+120, 60, 36)
	pygame.draw.rect(screen, (0,0,0), yes_rect)
	pygame.draw.rect(screen, (0,0,0), no_rect)
	yes_text = FONT_BUTTON.render("YES", True, (255,0,0))
	no_text = FONT_BUTTON.render("NO", True, (255,255,255))
	screen.blit(yes_text, yes_text.get_rect(center=yes_rect.center))
	screen.blit(no_text, no_text.get_rect(center=no_rect.center))

	# Puntuación
	score_text = FONT_SMALL.render(f"SCORE {score}", True, (255,255,255))
	score_rect = score_text.get_rect(center=(base_x, base_y-120))
	screen.blit(score_text, score_rect)

	return yes_rect, no_rect

def get_player_name(screen, fondo_surface=None):
    name = ""
    clock = pygame.time.Clock()
    input_active = True
    backspace_pressed = False
    backspace_timer = 0
    while input_active:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN and len(name) > 0:
                    input_active = False
                elif event.key == pygame.K_BACKSPACE:
                    backspace_pressed = True
                    name = name[:-1]
                elif len(name) < 10:  # Limitar a 10 caracteres
                    # Solo aceptar letras y números
                    if event.unicode.isalnum():
                        name += event.unicode
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_BACKSPACE:
                    backspace_pressed = False
                    backspace_timer = 0
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                if guardar_btn.collidepoint(mx, my) and len(name) > 0:
                    input_active = False

        # Manejar borrado continuo
        if backspace_pressed:
            backspace_timer += 1
            if backspace_timer > 10:  # Esperar un poco antes de empezar a borrar
                if backspace_timer % 5 == 0:  # Borrar cada 5 frames
                    name = name[:-1]

        if fondo_surface:
            screen.blit(fondo_surface, (0, 0))
        else:
            screen.fill((0, 0, 0))
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        overlay.set_alpha(180)
        overlay.fill((0, 0, 0))
        screen.blit(overlay, (0, 0))
        win_w, win_h = 540, 280  # Ventana más ancha y alta
        win_x = (SCREEN_WIDTH - win_w) // 2
        win_y = (SCREEN_HEIGHT - win_h) // 2
        pygame.draw.rect(screen, (40, 40, 60), (win_x, win_y, win_w, win_h), border_radius=16)
        pygame.draw.rect(screen, (255,255,255), (win_x, win_y, win_w, win_h), 3, border_radius=16)
        titulo = FONT_TITLE.render("Ingresa tu nombre", True, (255,255,0))
        screen.blit(titulo, (win_x + (win_w-titulo.get_width())//2, win_y+20))
        
        # Campo de texto
        input_rect = pygame.Rect(win_x+50, win_y+80, win_w-100, 40)
        pygame.draw.rect(screen, (255,255,255), input_rect, 2, border_radius=8)
        
        # Calcular el ancho máximo del texto (10 caracteres)
        max_width = FONT_BUTTON.render("W" * 10, True, (255,255,255)).get_width()
        # Calcular el ancho actual del texto
        name_surface = FONT_BUTTON.render(name, True, (255,255,255))
        current_width = name_surface.get_width()
        
        # Calcular la posición x para centrar el texto
        # Si el texto es más corto que el máximo, centrarlo
        if current_width < max_width:
            text_x = input_rect.centerx - current_width//2
        else:
            # Si el texto es más largo, alinearlo a la izquierda
            text_x = input_rect.x + 10
        
        text_y = input_rect.centery - name_surface.get_height()//2
        screen.blit(name_surface, (text_x, text_y))
        
        if len(name) < 10:
            # Posicionar el cursor después del texto
            cursor_x = text_x + name_surface.get_width()
            cursor_y = input_rect.centery - 10
            pygame.draw.line(screen, (255,255,255), (cursor_x, cursor_y), (cursor_x, cursor_y+20), 2)
        
        # Instrucciones justo debajo del campo de texto
        instrucciones = FONT_SMALL.render("Máximo 10 caracteres (solo letras y números)", True, (200,200,200))
        screen.blit(instrucciones, (win_x + (win_w-instrucciones.get_width())//2, win_y+140))
        
        # Botón Guardar más abajo
        guardar_btn = pygame.Rect(win_x+win_w//2-60, win_y+win_h-60, 120, 36)
        color_btn = (0,180,0) if len(name) > 0 else (120,120,120)
        pygame.draw.rect(screen, color_btn, guardar_btn, border_radius=8)
        pygame.draw.rect(screen, (0,0,0), guardar_btn, 2, border_radius=8)
        guardar_txt = FONT_BUTTON.render("Guardar", True, (255,255,255))
        screen.blit(guardar_txt, (guardar_btn.centerx-guardar_txt.get_width()//2, guardar_btn.centery-guardar_txt.get_height()//2))
        
        pygame.display.flip()
        clock.tick(60)
    return name

def run_game():
	screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
	pygame.display.set_caption("EcoRunner: Nave contra la Basura")
	clock = pygame.time.Clock()

	# Renderizar la pantalla de selección de dificultad como fondo para el modal de nombre
	bg = pygame.transform.smoothscale(menu_background, (SCREEN_WIDTH, SCREEN_HEIGHT))
	screen.blit(bg, (0, 0))
	draw_gradient_title(screen, "Selecciona la dificultad o modo de IA", -40)
	# Botones de dificultad (solo visual, sin interacción)
	button_data = [
		(button_imgs[0], "Fácil"),
		(button_imgs[1], "Normal"),
		(button_imgs[2], "Difícil"),
		(button_imgs[3], "IA")
	]
	button_width = 320
	button_height = 56
	button_gap = 28
	start_y = SCREEN_HEIGHT // 2 - 140
	for i, (img, label) in enumerate(button_data):
		rect = pygame.Rect(SCREEN_WIDTH//2 - button_width//2, start_y + i * (button_height + button_gap), button_width, button_height)
		draw_image_button(screen, rect, img, label, FONT_BIG, False)
	pygame.display.flip()
	fondo_surface = screen.copy()

	# Pedir nombre antes de seleccionar dificultad
	nombre_jugador = get_player_name(screen, fondo_surface)

	q_agent = QLearningAgent(actions=[0, 1, 2, 3])
	q_table_path = "data/q_table.pkl"
	if os.path.exists(q_table_path):
		q_agent.load(q_table_path)

	running = True
	while running:
		# Recibe la skin seleccionada
		result = show_difficulty_menu(screen)
		if result is False:
			return False
		if not result:
			return True
		if len(result) == 3:
			difficulty, ia_mode, skin_idx = result
		else:
			difficulty, ia_mode = result
			skin_idx = 0
		modo_ia = difficulty == "IA"
		auto_mode = ia_mode == "Automatica"

		if modo_ia:
			trash_speed = 5
			obstacle_speed = 5
		elif difficulty == "Fácil":
			trash_speed = 3
			obstacle_speed = 3
		elif difficulty == "Normal":
			trash_speed = 5
			obstacle_speed = 5
		else:
			trash_speed = 8
			obstacle_speed = 8

		def reset_game():
			player = Player(100, SCREEN_HEIGHT // 2, skin_idx)
			player.speed = 6 if modo_ia else 7
			return {
				"player": player,
				"trash_list": [],
				"obstacles": [],
				"score": 0,
				"missed_trash": 0,
				"dark_overlay_alpha": 0,
				"last_damage_time": 0,
				"game_over": False,
				"explosions": [],
				"last_state": None,
				"last_action": None,
				"moved_up": False,
				"moved_down": False,
				"shot": False,
				"consecutive_trash": 0,
				"shield_active": False,
				"shield_time": 0,
				"shield_alpha": 0,
				"cleaner_ray_ready": False,
				"cleaner_ray_active": False,
				"cleaner_ray_time": 0,
				"cleaner_ray_alpha": 0,
				"cleaner_ray_y": 0,
				"ray_offset": 0,
				"laser_sound_playing": False  # Control para el sonido del láser
			}

		state = reset_game()

		TRASH_SPAWN = pygame.USEREVENT + 1
		OBSTACLE_SPAWN = pygame.USEREVENT + 2
		pygame.time.set_timer(TRASH_SPAWN, 1000)
		pygame.time.set_timer(OBSTACLE_SPAWN, 2500)

		in_game = True
		while in_game:
			# Dibujar el fondo base
			screen.blit(menu_background, (0, 0))
			
			# Calcular la transparencia basada en el número de basuras perdidas (máximo 10)
			overlay_alpha = (state["missed_trash"] / 10) * 180  # 180 es el máximo alpha para el overlay
			background_alpha = (state["missed_trash"] / 10) * 255  # 255 es completamente opaco para el fondo
			
			# Si hay basuras perdidas, dibujar el overlay oscuro y el fondo alternativo
			if state["missed_trash"] > 0:
				# Dibujar el overlay oscuro
				overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
				overlay.fill((0, 0, 0))
				overlay.set_alpha(overlay_alpha)
				screen.blit(overlay, (0, 0))
				
				# Dibujar el fondo alternativo
				background2_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
				background2_surface.blit(menu_background2, (0, 0))
				background2_surface.set_alpha(background_alpha)
				screen.blit(background2_surface, (0, 0))

			keys = pygame.key.get_pressed()
			now = pygame.time.get_ticks()

			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					running = False
					return False

				if event.type == pygame.MOUSEBUTTONDOWN:
					mx, my = pygame.mouse.get_pos()
					if not state["game_over"]:
						menu_btn = draw_menu_button(screen, pygame.mouse.get_pos())
						if menu_btn.collidepoint(mx, my):
							in_game = False
						elif event.button == 1 and not modo_ia:  # Click izquierdo
							# Solo reproducir el sonido si realmente se crea una bala
							if state["player"].shoot():
								laser_sound.play()
								state["shot"] = True
						elif event.button == 3 and state["cleaner_ray_ready"]:  # Click derecho
							state["cleaner_ray_active"] = True
							state["cleaner_ray_time"] = 300  # 5 segundos a 60 FPS
							state["cleaner_ray_ready"] = False
							state["consecutive_trash"] = 0  # Reiniciar el contador de basuras
							state["cleaner_ray_y"] = state["player"].rect.centery
							# Iniciar el sonido del láser
							laser2_sound.play(-1)  # -1 para que se repita indefinidamente
							state["laser_sound_playing"] = True
					else:
						yes_rect, no_rect = draw_game_over_glitch(screen, state["score"])
						if yes_rect and yes_rect.collidepoint(mx, my):
							state = reset_game()
						elif no_rect and no_rect.collidepoint(mx, my):
							in_game = False

				if not state["game_over"]:
					if event.type == TRASH_SPAWN:
						y = random.randint(50, SCREEN_HEIGHT - 50)
						t = Trash(SCREEN_WIDTH + 20, y)
						t.speed = trash_speed
						state["trash_list"].append(t)
					if event.type == OBSTACLE_SPAWN:
						y = random.randint(50, SCREEN_HEIGHT - 50)
						o = Obstacle(SCREEN_WIDTH + 20, y)
						o.set_speed(obstacle_speed)
						state["obstacles"].append(o)

			if not state["game_over"]:
				# Actualizar estado del escudo
				if state["shield_active"]:
					state["shield_time"] -= 1
					if state["shield_time"] <= 0:
						state["shield_active"] = False
						state["shield_alpha"] = 0
					else:
						# Efecto pulsante del escudo
						state["shield_alpha"] = 128 + int(127 * math.sin(now / 200))

				# Actualizar estado del rayo limpiador
				if state["cleaner_ray_active"]:
					state["cleaner_ray_time"] -= 1
					if state["cleaner_ray_time"] <= 0:
						state["cleaner_ray_active"] = False
						state["cleaner_ray_alpha"] = 0
						# Detener el sonido del láser
						if state["laser_sound_playing"]:
							laser2_sound.stop()
							state["laser_sound_playing"] = False
					else:
						# Efecto pulsante del rayo
						state["cleaner_ray_alpha"] = 180 + int(75 * math.sin(now / 100))
						# Actualizar posición Y del rayo para que siga a la nave
						state["cleaner_ray_y"] = state["player"].rect.centery
						# Actualizar offset para el efecto de movimiento
						state["ray_offset"] = (state["ray_offset"] + 5) % 20

				# Dibujar los mensajes de ayuda en la parte superior
				help_font = FONT_SMALL
				help_color = (255, 255, 255)
				help_shadow = (0, 0, 0)
				y_position = 50  # Posición fija en la parte superior

				if not state["moved_up"]:
					text = "↑ W: Mover arriba"
					shadow = help_font.render(text, True, help_shadow)
					text_surface = help_font.render(text, True, help_color)
					shadow_rect = shadow.get_rect(center=(SCREEN_WIDTH//2+1, y_position+1))
					text_rect = text_surface.get_rect(center=(SCREEN_WIDTH//2, y_position))
					screen.blit(shadow, shadow_rect)
					screen.blit(text_surface, text_rect)

				if not state["moved_down"]:
					text = "↓ S: Mover abajo"
					shadow = help_font.render(text, True, help_shadow)
					text_surface = help_font.render(text, True, help_color)
					shadow_rect = shadow.get_rect(center=(SCREEN_WIDTH//2+1, y_position+30+1))
					text_rect = text_surface.get_rect(center=(SCREEN_WIDTH//2, y_position+30))
					screen.blit(shadow, shadow_rect)
					screen.blit(text_surface, text_rect)

				if not state["shot"]:
					text = "Click Izquierdo: Disparar"
					shadow = help_font.render(text, True, help_shadow)
					text_surface = help_font.render(text, True, help_color)
					shadow_rect = shadow.get_rect(center=(SCREEN_WIDTH//2+1, y_position+60+1))
					text_rect = text_surface.get_rect(center=(SCREEN_WIDTH//2, y_position+60))
					screen.blit(shadow, shadow_rect)
					screen.blit(text_surface, text_rect)

				# Dibujar el botón de menú
				mouse_pos = pygame.mouse.get_pos()
				menu_btn = draw_menu_button(screen, mouse_pos)

				if modo_ia:
					trash_y = state["trash_list"][0].y if state["trash_list"] else SCREEN_HEIGHT // 2
					current_state = (state["player"].rect.y // 50, trash_y // 50)

					if state["last_state"] is not None:
						q_agent.learn(state["last_state"], state["last_action"], -1, current_state)

					action = q_agent.choose_action(current_state)
					state["last_state"] = current_state
					state["last_action"] = action

					if action == 1:
						state["player"].rect.y -= state["player"].speed
					elif action == 2:
						state["player"].rect.y += state["player"].speed
					elif action == 3:
						state["player"].shoot()
						laser_sound.play()
				else:
					# Actualizar estado de ayuda contextual
					if keys[pygame.K_w]:
						state["moved_up"] = True
					if keys[pygame.K_s]:
						state["moved_down"] = True
					state["player"].handle_keys(keys)

				state["player"].draw(screen)

				# Dibujar el escudo si está activo
				if state["shield_active"]:
					shield_surface = pygame.Surface((state["player"].rect.width + 40, state["player"].rect.height + 40), pygame.SRCALPHA)
					pygame.draw.ellipse(shield_surface, (0, 255, 255, state["shield_alpha"]), 
									 (0, 0, state["player"].rect.width + 40, state["player"].rect.height + 40), 3)
					screen.blit(shield_surface, 
							  (state["player"].rect.x - 20, state["player"].rect.y - 20))

				# Dibujar el rayo limpiador si está activo
				if state["cleaner_ray_active"]:
					# Crear el efecto del rayo
					ray_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
					
					# Dibujar el rayo principal con efecto de movimiento
					for i in range(3):
						offset = state["ray_offset"] + (i * 20)
						alpha = state["cleaner_ray_alpha"] - (i * 40)
						if alpha > 0:
							# Línea principal con efecto de movimiento
							pygame.draw.line(ray_surface, (0, 255, 255, alpha),
										  (state["player"].rect.right + offset, state["cleaner_ray_y"]),
										  (SCREEN_WIDTH, state["cleaner_ray_y"]), 5)
							
							# Efectos secundarios con variación
							for j in range(2):
								y_offset = (j - 0.5) * 15
								alpha_secondary = alpha // 2
								pygame.draw.line(ray_surface, (0, 255, 255, alpha_secondary),
											  (state["player"].rect.right + offset, state["cleaner_ray_y"] + y_offset),
											  (SCREEN_WIDTH, state["cleaner_ray_y"] + y_offset), 3)
					
					# Añadir partículas brillantes
					for _ in range(5):
						x = random.randint(state["player"].rect.right, SCREEN_WIDTH)
						y = state["cleaner_ray_y"] + random.randint(-20, 20)
						size = random.randint(2, 4)
						alpha = random.randint(100, 200)
						pygame.draw.circle(ray_surface, (0, 255, 255, alpha), (x, y), size)
					
					screen.blit(ray_surface, (0, 0))

				for trash in state["trash_list"][:]:
					trash.move()
					trash.draw(screen)
					trash_removed = False

					# Si el rayo limpiador está activo, destruir la basura
					if state["cleaner_ray_active"] and abs(trash.y - state["cleaner_ray_y"]) < 50:
						explosion = Explosion(trash.x - trash.radius, trash.y - trash.radius, trash.radius * 2)
						state["explosions"].append(explosion)
						explosion_sound.play()
						state["trash_list"].remove(trash)
						state["score"] += 15  # Puntos extra por destruir con el rayo
						trash_removed = True
						continue

					if trash.x + trash.radius < 0:
						state["trash_list"].remove(trash)
						state["missed_trash"] = min(state["missed_trash"] + 1, 10)
						state["consecutive_trash"] = 0
						trash_removed = True
						continue

					for proj in state["player"].projectiles:
						if trash.rect().colliderect(proj):
							explosion = Explosion(trash.x - trash.radius, trash.y - trash.radius, trash.radius * 2)
							state["explosions"].append(explosion)
							explosion_sound.play()

							if modo_ia:
								q_agent.learn(state["last_state"], state["last_action"], 10, state["last_state"])

							if not trash_removed:
								state["trash_list"].remove(trash)
								trash_removed = True
							if proj in state["player"].projectiles:
								state["player"].projectiles.remove(proj)

							state["score"] += 10
							state["missed_trash"] = max(0, state["missed_trash"] - 1)
							state["consecutive_trash"] += 1
							
							# Activar escudo si se recolectan 5 basuras consecutivas
							if state["consecutive_trash"] >= 5 and not state["shield_active"]:
								state["shield_active"] = True
								state["shield_time"] = 300  # 5 segundos a 60 FPS
							
							# Cargar rayo limpiador si se recolectan 10 basuras consecutivas
							if state["consecutive_trash"] >= 10:
								state["cleaner_ray_ready"] = True

							state["missed_trash"] = max(0, state["missed_trash"] - 1)
							break

					if not trash_removed and trash.rect().colliderect(state["player"].rect):
						state["trash_list"].remove(trash)
						explosion_sound.play()
						state["player"].health -= 10
						if state["player"].health <= 0:
							state["game_over"] = True
							# Desactivar el rayo si está activo
							if state["cleaner_ray_active"]:
								state["cleaner_ray_active"] = False
								state["cleaner_ray_alpha"] = 0
								if state["laser_sound_playing"]:
									laser2_sound.stop()
									state["laser_sound_playing"] = False

				for obs in state["obstacles"][:]:
					obs.move()
					obs.draw(screen)
					if obs.rect.colliderect(state["player"].rect):
						if not state["shield_active"]:  # Solo dañar si no hay escudo
							explosion_sound.play()
							state["game_over"] = True
							# Desactivar el rayo si está activo
							if state["cleaner_ray_active"]:
								state["cleaner_ray_active"] = False
								state["cleaner_ray_alpha"] = 0
								if state["laser_sound_playing"]:
									laser2_sound.stop()
									state["laser_sound_playing"] = False
						else:
							# Hacer rebotar el pájaro hacia abajo-izquierda
							obs.speed_y = obs.speed  # Rebotar hacia abajo
							obs.speed_x = -obs.speed  # Mantener dirección hacia la izquierda
							
							# Reproducir sonido de rebote
							explosion_sound.play()
							
							# Añadir un pequeño efecto visual de rebote
							explosion = Explosion(obs.rect.x, obs.rect.y, obs.rect.width // 2)
							state["explosions"].append(explosion)
							
							# Eliminar el pájaro si sale de la pantalla
							if obs.x > SCREEN_WIDTH or obs.x < 0 or obs.y < 0 or obs.y > SCREEN_HEIGHT:
								state["obstacles"].remove(obs)

				if state["missed_trash"] >= 10 and now - state["last_damage_time"] >= 5000:
					state["player"].health -= 10
					state["last_damage_time"] = now
					if state["player"].health <= 0:
						state["game_over"] = True

				draw_game_ui(screen, state["score"], state["player"].health, state["consecutive_trash"], state["cleaner_ray_ready"])

				for explosion in state["explosions"][:]:
					explosion.update()
					explosion.draw(screen)
					if explosion.finished:
						state["explosions"].remove(explosion)

			else:
				if modo_ia:
					q_agent.learn(state["last_state"], state["last_action"], -10, state["last_state"])
					q_agent.save(q_table_path)

				if not state.get("score_saved", False):
					with open("data/scores.csv", "a") as f:
						f.write(f"{nombre_jugador},{difficulty},{state['score']}\n")
					state["score_saved"] = True

				yes_rect, no_rect = draw_game_over_glitch(screen, state["score"])

			pygame.display.flip()
			clock.tick(FPS)

	return True
