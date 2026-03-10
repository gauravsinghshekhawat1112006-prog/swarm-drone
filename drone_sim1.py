import pygame
import math
import numpy as np

pygame.init()

WIDTH = 800
HEIGHT = 600

drone_x = 400
drone_y = 300

desired_x = 600
desired_y = 500

# vx = 0.01
# vy = 0.01

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Drone Simulation")

clock = pygame.time.Clock()

running = True

while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((30,30,30))
    pygame.draw.circle(screen,(255,0,0),(desired_x,desired_y),8)
    pygame.draw.circle(screen,(0,255,0),(float(drone_x),float(drone_y)),10)

    error_x = desired_x - drone_x
    vx = 0.05*error_x
    error_y = desired_y - drone_y
    vy = 0.05*error_y

    drone_x += vx
    drone_y += vy

    # drone_x += 0.01
    # drone_y += math.sin(drone_x)

    pygame.display.update()
    clock.tick(60)

pygame.quit()