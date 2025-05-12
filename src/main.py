import pygame
import sys
from game_loop import run_game

if __name__ == "__main__":
    pygame.init()
    while True:
        seguir = run_game()
        if seguir is False:
            pygame.quit()
            sys.exit()
    

