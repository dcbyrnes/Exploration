import pygame
import random
import numpy as np
import os
from gym_maze.envs.maze_view_2d import Maze

maze_file_path = '/Users/user/Code/gym-maze/gym_maze/envs/maze_samples/milestone_maze.npy'
screen_size = (600, 600)
has_loops = True
num_portals = 3

pygame.init()
pygame.display.set_caption('milestone maze')

maze = Maze(maze_cells=Maze.load_maze(maze_file_path), has_loops=has_loops, num_portals=num_portals)

maze_size = maze.maze_size
        # to show the right and bottom border
screen = pygame.display.set_mode(screen_size)
screen_size = tuple(map(sum, zip(screen_size, (-1, -1))))
entrance = np.zeros(2, dtype=int)
goal = np.array(maze_size) - np.array((1, 1))
robot = entrance
background = pygame.Surface(screen.get_size()).convert()
background.fill((255, 255, 255))
maze_layer = pygame.Surface(screen.get_size()).convert_alpha()
maze_layer.fill((0, 0, 0, 0,))
line_colour = (0, 0, 0, 255)
SCREEN_SIZE = tuple(screen_size)
SCREEN_W = int(SCREEN_SIZE[0])
SCREEN_H = int(SCREEN_SIZE[1])
CELL_W = float(SCREEN_W) / float(maze.MAZE_W)
CELL_H = float(SCREEN_H) / float(maze.MAZE_H)

#################
#### DRAW MAZE
#################

# drawing the horizontal lines
for y in range(maze.MAZE_H + 1):
    pygame.draw.line(maze_layer, line_colour, (0, y * CELL_H),
                     (SCREEN_W, y * CELL_H), 3)

# drawing the vertical lines
for x in range(maze.MAZE_W + 1):
    pygame.draw.line(maze_layer, line_colour, (x * CELL_W, 0),
                     (x * CELL_W, SCREEN_H), 3)

def cover_walls(x, y, dirs, colour=(0, 0, 255, 15)):

    dx = x * CELL_W
    dy = y * CELL_H

    if not isinstance(dirs, str):
        raise TypeError("dirs must be a str.")

    for dir in dirs:
        if dir == "S":
            line_head = (dx + 1, dy + CELL_H)
            line_tail = (dx + CELL_W - 1, dy + CELL_H)
        elif dir == "N":
            line_head = (dx + 1, dy)
            line_tail = (dx + CELL_W - 1, dy)
        elif dir == "W":
            line_head = (dx, dy + 1)
            line_tail = (dx, dy + CELL_H - 1)
        elif dir == "E":
            line_head = (dx + CELL_W, dy + 1)
            line_tail = (dx + CELL_W, dy + CELL_H - 1)
        else:
            raise ValueError("The only valid directions are (N, S, E, W).")

        pygame.draw.line(maze_layer, colour, line_head, line_tail, 3)

# breaking the walls
for x in range(len(maze.maze_cells)):
    for y in range (len(maze.maze_cells[x])):
        # check the which walls are open in each cell
        walls_status = maze.get_walls_status(maze.maze_cells[x, y])
        dirs = ""
        for dir, open in walls_status.items():
            if open:
                dirs += dir
        print(x,y, dirs)
        cover_walls(x, y, dirs)

#################
#### DRAW PORTALS
#################
def colour_cell(cell, colour, transparency):

    if not (isinstance(cell, (list, tuple, np.ndarray)) and len(cell) == 2):
        raise TypeError("cell must a be a tuple, list, or numpy array of size 2")

    x = int(cell[0] * CELL_W + 0.5 + 1)
    y = int(cell[1] * CELL_H + 0.5 + 1)
    w = int(CELL_W + 0.5 - 1)
    h = int(CELL_H + 0.5 - 1)
    pygame.draw.rect(maze_layer, colour + (transparency,), (x, y, w, h))

transparency = 255
colour_range = np.linspace(0, 255, len(maze.portals), dtype=int)
colour_i = 0
for portal in maze.portals:
    colour = ((100 - colour_range[colour_i])% 255, colour_range[colour_i], 0)
    colour_i += 1
    for location in portal.locations:
        colour_cell(location, colour=colour, transparency=transparency)

#################
#### DRAW ENTRANCE
#################

colour_cell((0,0), colour=(0,0,255), transparency=transparency)

#################
#### DRAW GOAL
#################

colour_cell((9,9), colour=(255,0,0), transparency=transparency)

#################
#### DRAW ROBOTS
#################
def draw_robot(x,y,colour=(50, 50, 50), transparency=100):

   x = int(x * CELL_W + CELL_W * 0.5 + 0.5)
   y = int(y * CELL_H + CELL_H * 0.5 + 0.5)
   r = int(min(CELL_W, CELL_H)/5 + 0.5)

   pygame.draw.circle(maze_layer, colour + (transparency,), (x, y), r)

path = [
# (0,0),
(1,0),(2,0),(3,0),(4,0),(5,0),(6,0),(6,1),(5,1),(5,2),
    # (5,3),(3,4),
    (3,5),
    # (3,6),(8,4),
    (9,4),(9,5),(9,6),(9,7),(9,8)
    # ,(9,9)
    ]
for x,y in path:
    draw_robot(x,y)

# draw_robot(5,3,colour=(40,40,0), transparency=200)
# draw_robot(3,4,colour=(40,40,0), transparency=200)

# draw_robot(5,3,colour=(0,40,40), transparency=200)
# draw_robot(3,4,colour=(0,40,40), transparency=200)



screen.blit(background, (0, 0))
screen.blit(maze_layer,(0, 0))

pygame.image.save(screen, 'foofoo.png')
