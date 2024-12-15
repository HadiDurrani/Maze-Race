import random
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

def create_grid(width, height):
    """Create a grid with walls everywhere."""
    return np.zeros((height * 2 + 1, width * 2 + 1), dtype=int)

def add_frontier(frontier, x, y, maze):
    """Add frontier cells to the list."""
    for nx, ny in [(x - 2, y), (x + 2, y), (x, y - 2), (x, y + 2)]:
        if 0 < nx < maze.shape[1] - 1 and 0 < ny < maze.shape[0] - 1 and maze[ny, nx] == 0:
            if (nx, ny) not in frontier:
                frontier.append((nx, ny))
                maze[ny, nx] = 2 

def neighbors(x, y, maze):
    """Find the neighbors of a cell that are part of the maze."""
    n = []
    for nx, ny in [(x - 2, y), (x + 2, y), (x, y - 2), (x, y + 2)]:
        if 0 < nx < maze.shape[1] - 1 and 0 < ny < maze.shape[0] - 1 and maze[ny, nx] == 1:
            n.append((nx, ny))
    return n

def visualize_maze(maze, ax, entry=None, exit=None, pause_time=0.01):
    ax.clear()
    cmap = mcolors.ListedColormap(['black', 'white'])
    bounds = [0, 1, 2]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    ax.imshow(maze, cmap=cmap, norm=norm, origin="upper")
    ax.axis("off")

    if entry:
        ax.scatter(entry[1], entry[0], color="yellow", s=100, label="Entry")  # Yellow for entry
    if exit:
        ax.scatter(exit[1], exit[0], color="red", s=100, label="Exit")  # Red for exit

    plt.pause(pause_time)

def prim_maze(width, height, visualize=True, loop_factor=0.1):
    """Generate a maze using Prim's algorithm with optional loops."""
    maze = create_grid(width, height)
    frontier = []

    start_x, start_y = random.randint(0, width - 1) * 2 + 1, random.randint(0, height - 1) * 2 + 1
    maze[start_y, start_x] = 1 
    add_frontier(frontier, start_x, start_y, maze)

    if visualize:
        fig, ax = plt.subplots(figsize=(10, 10))
        visualize_maze(maze, ax)

    while frontier:
        fx, fy = random.choice(frontier)
        frontier.remove((fx, fy))

        maze_neighbors = neighbors(fx, fy, maze)
        if maze_neighbors:
            nx, ny = random.choice(maze_neighbors)

            maze[(fy + ny) // 2, (fx + nx) // 2] = 1
            maze[fy, fx] = 1

            add_frontier(frontier, fx, fy, maze)

        if visualize:
            visualize_maze(maze, ax)

    introduce_loops(maze, loop_factor)
    plt.show()
    return maze

def introduce_loops(maze, loop_factor):
    #Randomly remove walls to create loops in the maze.
    height, width = maze.shape
    potential_walls = []

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if maze[y, x] == 0:
                if (maze[y - 1, x] == 1 and maze[y + 1, x] == 1) or (maze[y, x - 1] == 1 and maze[y, x + 1] == 1):
                    potential_walls.append((y, x))

    num_loops = int(len(potential_walls) * loop_factor)
    for _ in range(num_loops):
        y, x = random.choice(potential_walls)
        maze[y, x] = 1
        potential_walls.remove((y, x))

def add_entry_exit(maze):
    height, width = maze.shape

    for x in range(1, width, 2):  
        if maze[1, x] == 1:  
            maze[0, x] = 1  
            entry_x = x
            break

    for x in range(width - 2, 0, -2):  
        if maze[height - 2, x] == 1:  
            maze[height - 1, x] = 1 
            exit_x = x
            break

    return (0, entry_x), (height - 1, exit_x)  

if __name__ == "__main__":
    width, height = 20,20
    maze = prim_maze(width, height, visualize=True, loop_factor=0.15)

    entry, exit = add_entry_exit(maze)

    fig, ax = plt.subplots(figsize=(10, 10))
    visualize_maze(maze, ax, entry=entry, exit=exit, pause_time=0.01)

    plt.title("Maze with Entry (Yellow) and Exit (Red)")
    plt.legend()
    plt.show()

    # Save the maze to a file
    np.save("saved_maze.npy", maze)
    print("Maze saved to 'saved_maze.npy'")
