import random
import os
import time
from typing import NewType, Any, List, Tuple, Set, Dict

import yaml
import pygame
import numpy as np


 # A Cell is represented by a (column, row) pair
Cell = NewType('Cell', Tuple[int, int])

class GameOfLife:
    """
    Implementation of the Conway's Game of Life using Pygame.

    ATTRIBUTES:
        - _config: configuration dictionary (from 'config.yml')
        - _n_cols: number of columns in the grid
        - _n_rows: number of rows in the grid
        - _n_cells: total number of cells in the grid (_n_cols * _n_rows)
        - _screen: pygame surface where the simulation is displayed
        - clock: pygame clock to control the speed of the simulation
        - _living_cells: set of living cells {(col, row)} in current timestep
        - _is_paused: whether the simulation is running or frozen
        - _run_next_step: whether to run just the next step (only if _is_paused)
        - _current_pattern: Pattern seed id or 'Rand' if random pattern

    METHODS:
        - process_events: Translate mouse clicks and keyboard actions (from
            the user) to specific events in the simulation.
        - run_logic:  Update the grid following the Rules of the game.
        - draw: Display the elements of the simulations on '_screen' attribute.
    """

    def __init__(self) -> None:
        """
        Initialize a GameOfLife instance
        """
        # Read the configuration file
        self._config = self._get_config()

        # Set the random seed if provided
        if self._config['random_seed'] is not None:
            random.seed(self._config['random_seed'])

        # Initialize Pygame
        pygame.init()
        self._screen = pygame.display.set_mode(
            (self._config['width'], self._config['height']),
            pygame.RESIZABLE  # Allow the window to be resizable
        )
        self.clock = pygame.time.Clock()

        # Initialize grid dimensions
        self._update_grid_dimensions(self._config['width'], self._config['height'])

        # Simulation elements
        self._living_cells = set()
        self._is_paused = True
        self._run_next_step = False
        self._current_pattern = None
        self._generation = 0

        # Store the current window title
        self._current_title = ""

    def _update_grid_dimensions(self, width: int, height: int) -> None:
        """
        Update the grid dimensions based on the window size.

        :param width: New width of the window
        :param height: New height of the window
        """
        self._config['width'] = width
        self._config['height'] = height
        self._n_cols = width // self._config['cell_size']
        self._n_rows = height // self._config['cell_size']
        self._n_cells = self._n_rows * self._n_cols

    @staticmethod
    def _get_config() -> Dict[str, Any]:
        """
        Read the configuration file called 'config.yml' and return it as a
        python dictionary.

        :return: configuration dictionary
        """

        this_file_path = os.path.abspath(__file__)
        project_path = os.path.dirname(this_file_path)
        config_path = os.path.join(project_path, 'config.yml')

        with open(config_path, 'r') as yml_file:
            config = yaml.safe_load(yml_file)[0]['config']
        return config


    def _generate_random_init_grid(self) -> Set[Cell]:
        """
        Generate an initial random pattern to start the simulation.

        :return: set of living cells representing the initial pattern
        """

        self._current_pattern = 'Rand'
        # Random percentage of living cells. Max and Min values are in config.
        pct_living_cells = random.randrange(
            start=self._config['gen_min_pct_living_cells'],
            stop=self._config['gen_max_pct_living_cells']
        )
        new_living_cells = set()  # init living cells
        n_cells_to_gen = (self._n_cells * pct_living_cells) // 100
        for _ in range(n_cells_to_gen):
            row = random.randrange(start=0, stop=self._n_rows)
            col = random.randrange(start=0, stop=self._n_cols)
            new_living_cells.add(Cell((col, row)))
        return new_living_cells


    def _generate_seed_pattern(self, id_: int) -> Set[Cell]:
        """
        Load the initial pattern with the given 'id_' from the file
        'seed_patterns.yml'. There are 9 different 'id_' values.
        The patterns are centered to the middle of the grid.

        :param id_: identifier of one of the available patterns (from 1 to 9)
        :raise: ValueError if the given 'id_' is not valid
        :return: set of initial living cells, setting up the pattern
        """

        if not (1 <= id_ <= 9):
            raise ValueError("the given pattern 'id_' must be between 1 and 9")

        this_file_path = os.path.abspath(__file__)
        project_path = os.path.dirname(this_file_path)
        seed_patterns_path = os.path.join(project_path, 'seed_patterns.yml')

        try:
            with open(seed_patterns_path, 'r') as yml_file:
                binary_pattern = yaml.safe_load(yml_file)[0]['patterns'][id_]
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {seed_patterns_path}")
        except KeyError:
            raise KeyError(f"Pattern ID {id_} not found in seed_patterns.yml")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")

        # Create a binary two-dimensional array {zeros: dead; ones: living}
        binary_pattern = np.array(binary_pattern)
        # Compute the top-left corner to place the pattern in the center
        top_left_col = (self._n_cols - len(binary_pattern[0])) // 2
        top_left_row = (self._n_rows - len(binary_pattern)) // 2

        # Get the Cells [(col, row)] with ones (living cells)
        # Pair the two tuples from the 'where' funtion (lists rows and cols)
        seed_pattern_living_cells = zip(*np.where(binary_pattern))
        pattern_living_cells = set()
        for row, col in seed_pattern_living_cells:
            pattern_living_cells.add(
                Cell((col+top_left_col, row+top_left_row))
            )
        return pattern_living_cells


    def process_events(self) -> bool:
        """
        Process user input events.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.VIDEORESIZE:
                # Handle window resizing
                new_width, new_height = event.w, event.h
                self._screen = pygame.display.set_mode((new_width, new_height), pygame.RESIZABLE)
                self._update_grid_dimensions(new_width, new_height)

            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                # Check if "Zoom In" button is clicked
                if 10 <= x <= 110 and 10 <= y <= 50:
                    self._config['cell_size'] = min(self._config['cell_size'] + 1, 50)  # Max cell size
                    self._update_grid_dimensions(self._config['width'], self._config['height'])
                # Check if "Zoom Out" button is clicked
                elif 120 <= x <= 220 and 10 <= y <= 50:
                    self._config['cell_size'] = max(self._config['cell_size'] - 1, 5)  # Min cell size
                    self._update_grid_dimensions(self._config['width'], self._config['height'])

                # Toggle cell state (alive/dead)
                col = x // self._config['cell_size']
                row = y // self._config['cell_size']
                cell = Cell((col, row))
                if cell in self._living_cells:
                    self._living_cells.remove(cell)
                else:
                    self._living_cells.add(cell)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self._is_paused = not self._is_paused
                elif event.key == pygame.K_RIGHT and self._is_paused:
                    self._run_next_step = True
                elif event.key == pygame.K_c:
                    # Clear the grid and reset the generation counter
                    self._living_cells.clear()
                    self._is_paused = True
                    self._generation = 0  # Reset generation counter
                elif event.key == pygame.K_g:
                    self._living_cells = self._generate_random_init_grid()
                    self._is_paused, self._current_pattern = True, 'Rand'
                elif pygame.K_1 <= event.key <= pygame.K_9:
                    pattern_id = event.key - pygame.K_0
                    self._living_cells = self._generate_seed_pattern(id_=pattern_id)
                    self._is_paused, self._current_pattern = True, str(pattern_id)
        return True


    def run_logic(self) -> None:
        """
        Update the grid following the Rules of the game.
        """
        if self._is_paused and not self._run_next_step:
            self._update_window_title()  # Update title even when paused
            return  # Do nothing, wait until the simulation is resumed

        if self._config['sleep'] is not None:  # Slow down the simulation
            time.sleep(self._config['sleep'])

        # Set containing all the neighbors of the currently living cells
        all_neighbors = set()
        # Set of the next generation cells, the next '_living_cells'
        new_living_cells = set()

        # For each living cell, get the neighbors and check if the cell will
        # live on to the next generation (survive).
        for cell in self._living_cells:
            cell_neighbors = self._get_neighbors(cell=cell)
            all_neighbors.update(cell_neighbors)
            cell_living_neighbors = list(
                filter(
                    lambda cell_: cell_ in self._living_cells, cell_neighbors
                )
            )
            if (self._config['underpopulation'] <=
                    len(cell_living_neighbors) <=
                    self._config['overpopulation']):
                new_living_cells.add(cell)
            # ELSE: the cell dies by underpopulation or overpopulation

        # For each neighbor of the currently living cells, check if it will be
        # brought back to life (reproduction)
        for cell in all_neighbors:
            cell_neighbors = self._get_neighbors(cell=cell)
            cell_living_neighbors = list(
                filter(
                    lambda cell_: cell_ in self._living_cells, cell_neighbors
                )
            )
            if len(cell_living_neighbors) == self._config['reproduction']:
                new_living_cells.add(cell)

        # Update the '_living_cells' attribute with the new generation
        self._living_cells = new_living_cells

        # Increment generation counter for both running and single-step modes
        if not self._is_paused or self._run_next_step:
            self._generation += 1

        # Reset the single-step flag
        self._run_next_step = False

        # Update the window title
        self._update_window_title()


    def _get_neighbors(self, cell: Cell) -> List[Cell]:
        """
        Returns the list of (at most) 8 neighbor of the given 'cell'.
        Each cell is represented by the pair (col, row).
        The grid is either infinite (unbounded) or finite (bounded),
        [See config].

        :param cell: (col, row)
        :return: set of neighbors of the given 'cell'
        """

        col, row = cell
        grid_is_infinite = self._config['grid_is_infinite']
        delta_row_vals, delta_col_vals = [-1, 0, 1], [-1, 0, 1]

        # Update the delta values if the cell is on the edge of the grid
        if row == self._n_rows - 1:  # Bottom row
            if grid_is_infinite:
                delta_row_vals[-1] = -self._n_rows + 1
            else:
                delta_row_vals = delta_row_vals[:-1]  # Remove last value

        elif row == 0:  # Top row
            if grid_is_infinite:
                delta_row_vals[0] = self._n_rows - 1
            else:
                delta_row_vals = delta_row_vals[1:]  # Remove first value

        if col == self._n_cols - 1:  # Rightmost column
            if grid_is_infinite:
                delta_col_vals[-1] = -self._n_cols + 1
            else:
                delta_col_vals = delta_col_vals[:-1]  # Remove last value

        elif col == 0:  # Leftmost column
            if grid_is_infinite:
                delta_col_vals[0] = self._n_cols - 1
            else:
                delta_col_vals = delta_col_vals[1:]  # Remove first value

        neighbors = []  # neighbors of the given 'cell' (living or dead)
        for delta_col in delta_col_vals:
            for delta_row in delta_row_vals:
                if delta_col == 0 and delta_row == 0:
                    continue  # this iteration is the given 'cell' itself
                neighbors.append(Cell((col+delta_col, row+delta_row)))
        return neighbors


    def draw(self) -> None:
        """
        Display the simulation elements on the screen (pygame surface).
        NOTE that the colors can be customized in the configuration file.
        Feel free to test different colors

        :return: None. Updates the screen content.
        """
        self._screen.fill(self._config['dead_cell_color'])  # background
        cell_size = self._config['cell_size']

        # Draw the living cells (using a different color)
        for col, row in self._living_cells:
            pygame.draw.rect(
                surface=self._screen,
                color=self._config['living_cell_color'],
                rect=(col * cell_size, row * cell_size, cell_size, cell_size)
            )

        # Draw the horizontal lines of the grid
        for row in range(self._n_rows):
            pygame.draw.line(
                surface=self._screen,
                color=self._config['grid_line_color'],
                start_pos=(0, row * cell_size),
                end_pos=(self._config['width'], row * cell_size)
            )

        # Draw the vertical lines of the grid
        for col in range(self._n_cols):
            pygame.draw.line(
                surface=self._screen,
                color=self._config['grid_line_color'],
                start_pos=(col * cell_size, 0),
                end_pos=(col * cell_size, self._config['height'])
            )

        # Draw zoom buttons
        self._draw_button("Zoom In", 10, 10, 100, 40, (0, 200, 0))
        self._draw_button("Zoom Out", 120, 10, 100, 40, (200, 0, 0))

        pygame.display.update()  # Update the content of the screen

    def _update_window_title(self) -> None:
        """
        Update the window title only if it has changed.
        """
        new_title = self._config['screen_caption'].format(
            pat=self._current_pattern,
            paused='paused' if self._is_paused else 'running'
        ) + f" | Generation: {self._generation}"

        if new_title != self._current_title:
            pygame.display.set_caption(new_title)
            self._current_title = new_title

    def _draw_button(self, text: str, x: int, y: int, width: int, height: int, color: Tuple[int, int, int]) -> None:
        """
        Draw a button on the screen.

        :param text: Text to display on the button
        :param x: X-coordinate of the button
        :param y: Y-coordinate of the button
        :param width: Width of the button
        :param height: Height of the button
        :param color: Color of the button (RGB tuple)
        """
        pygame.draw.rect(self._screen, color, (x, y, width, height))
        font = pygame.font.Font(None, 24)
        text_surface = font.render(text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=(x + width // 2, y + height // 2))
        self._screen.blit(text_surface, text_rect)


if __name__ == '__main__':
    simulation = GameOfLife()

    running = True
    while running:
        running = simulation.process_events()
        simulation.run_logic()
        simulation.draw()

        # Control the simulation speed (frames per second)
        simulation.clock.tick(simulation._config.get('fps', 30))

    pygame.quit()
