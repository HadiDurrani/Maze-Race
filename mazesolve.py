import numpy as np
import matplotlib.pyplot as plt
import heapq
from collections import deque
import sys

def load_maze(file_path):
    return np.load(file_path)

def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

def get_neighbors(maze, current):
    neighbors = []
    x, y = current
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1] and maze[nx, ny] == 1:
            neighbors.append((nx, ny))
    return neighbors

def visualize_solution(maze, path, start, goal, title):
    plt.figure(figsize=(10, 10))
    plt.imshow(maze, cmap='gray', origin='upper')
    path_x, path_y = zip(*path)
    plt.plot(path_y, path_x, color='red', linewidth=2, label="Solution Path")
    plt.scatter(start[1], start[0], color='yellow', s=100, label="Entry")
    plt.scatter(goal[1], goal[0], color='green', s=100, label="Exit")
    plt.legend()
    plt.axis("off")
    plt.title(title)
    plt.show()

import time

def visualize_current_state_with_timer(ax, maze, start, goal, came_from, current, start_time, pause_time=0.01):
    ax.clear()
    ax.imshow(maze, cmap="grey", origin="upper")
    current_path = []
    node = current
    while node in came_from:
        current_path.append(node)
        node = came_from[node]
    current_path.append(start)
    current_path.reverse()
    if current_path:
        path_x, path_y = zip(*current_path)
        ax.plot(path_y, path_x, color="red", linewidth=2)
    ax.scatter(start[1], start[0], color="yellow", s=100, label="Entry")
    ax.scatter(goal[1], goal[0], color="green", s=100, label="Exit")
    ax.legend()
    ax.axis("off")
    
    elapsed_time = time.time() - start_time
    ax.text(0, -2, f"Timer: {elapsed_time:.2f} seconds", color="blue", fontsize=12, ha="left")
    
    plt.pause(pause_time)

# A* Algorithm
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star(maze, start, goal):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {cell: float("inf") for cell in np.ndindex(maze.shape)}
    g_score[start] = 0
    f_score = {cell: float("inf") for cell in np.ndindex(maze.shape)}
    f_score[start] = heuristic(start, goal)

    fig, ax = plt.subplots(figsize=(6,6))
    start_time = time.time() 
    while open_set:
        _, current = heapq.heappop(open_set)
        visualize_current_state_with_timer(ax, maze, start, goal, came_from, current, start_time)
        if current == goal:
            return reconstruct_path(came_from, current)
        for neighbor in get_neighbors(maze, current):
            temp_g_score = g_score[current] + 1
            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return None

# Dijkstra
def dijkstra(maze, start, goal):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {cell: float("inf") for cell in np.ndindex(maze.shape)}
    g_score[start] = 0

    fig, ax = plt.subplots(figsize=(6,6))
    start_time = time.time() 
    while open_set:
        current_cost, current = heapq.heappop(open_set)
        visualize_current_state_with_timer(ax, maze, start, goal, came_from, current, start_time)
        if current == goal:
            return reconstruct_path(came_from, current)
        for neighbor in get_neighbors(maze, current):
            temp_g_score = g_score[current] + 1
            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                heapq.heappush(open_set, (g_score[neighbor], neighbor))
    return None

# BFS 
def bfs(maze, start, goal):
    queue = deque([start])
    came_from = {}
    visited = set()
    visited.add(start)

    fig, ax = plt.subplots(figsize=(6,6))
    start_time = time.time() 
    while queue:
        current = queue.popleft()
        visualize_current_state_with_timer(ax, maze, start, goal, came_from, current, start_time)
        if current == goal:
            return reconstruct_path(came_from, current)
        for neighbor in get_neighbors(maze, current):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                queue.append(neighbor)
    return None

# DFS 
def dfs(maze, start, goal):
    stack = [start]
    came_from = {}
    visited = set()

    fig, ax = plt.subplots(figsize=(6,6))
    start_time = time.time() 
    while stack:
        current = stack.pop()
        visualize_current_state_with_timer(ax, maze, start, goal, came_from, current, start_time)
        if current == goal:
            return reconstruct_path(came_from, current)
        if current not in visited:
            visited.add(current)
            for neighbor in get_neighbors(maze, current):
                if neighbor not in visited:
                    came_from[neighbor] = current
                    stack.append(neighbor)
    return None

def main():
    maze = load_maze("saved_maze.npy")
    start = (1, np.where(maze[0] == 1)[0][0])
    goal = (maze.shape[0] - 2, np.where(maze[-1] == 1)[0][0])
    print("Choose an algorithm to solve the maze:")
    print("1. A* Algorithm")
    print("2. Dijkstra's Algorithm")
    print("3. BFS (Breadth-First Search)")
    print("4. DFS (Depth-First Search)")
    choice = input("Enter your choice (1-4): ")
    if choice == '1':
        path = a_star(maze, start, goal)
        title = "A* Algorithm Solution"
    elif choice == '2':
        path = dijkstra(maze, start, goal)
        title = "Dijkstra's Algorithm Solution"
    elif choice == '3':
        path = bfs(maze, start, goal)
        title = "BFS Solution"
    elif choice == '4':
        path = dfs(maze, start, goal)
        title = "DFS Solution"
    else:
        print("Invalid choice!")
        sys.exit()
    if path:
        print("Path found")
        plt.show()
        #visualize_solution(maze, path, start, goal, title)
    else:
        print("No path found!")

if __name__ == "__main__":
    main()
