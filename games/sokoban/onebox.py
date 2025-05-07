import threading
import pygame
import time

from games.sokoban.sokoban import game_loop
from games.sokoban.sokoban_agent import main_loop

def simulate_move_handler(action):
    if action == "up":
        ev = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP)
    elif action == "down":
        ev = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_DOWN)
    elif action == "left":
        ev = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LEFT)
    elif action == "right":
        ev = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT)
    elif action == "R":
        ev = pygame.event.Event(pygame.KEYDOWN, {'unicode': 'r', 'key': 114, 'mod': 0, 'scancode': 21})
    elif action == "D":
        ev = pygame.event.Event(pygame.KEYDOWN, {'unicode': 'd', 'key': 100, 'mod': 0, 'scancode': 7})
    else:
        None
        # do nothing
        print(f"[WARNING] Invalid move: {action}")
    
    # Simulate a key press for the given move
    print('posting event ... ', ev)
    pygame.event.post(ev)

def agent():
    # hold for a second to make sure the game is ready.
    time.sleep(1)

    main_loop(
        api_provider = 'gemini',
        model_name  = 'gemini-2.5-flash-preview-04-17',
        num_threads = 1,
        thinking = True,
        modality = 'text-only',
        move_handler = simulate_move_handler,
    )

# start agent in another thread
threading.Thread(target = agent).start()

# the main game loop
game_loop(1, 'games/sokoban/simple_levels')

