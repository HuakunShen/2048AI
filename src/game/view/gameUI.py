import sys
import pygame
import numpy as np
from torch import Tensor
from typing import Union
from src.game.controller.game import Game
from src.game.utils import (
    KEY_MAP, K_q, K_r, UP, DOWN, LEFT, RIGHT, get_bg_color, BG_COLOR,
)


class GameUI(object):
    def __init__(self, matrix: Union[Tensor, np.ndarray] = None, game: Game = None, width: int = 800, height: int = 950,
                 margin: int = 10, fps: int = 30) -> None:
        """
        Init function for GameUI
        :param matrix: predefined 2D matrix for the game, if None then a random board will be generated
        :param game: Game: game object
        :param width: int: game board width
        :param height: int: game board height
        :param margin: margin of game screen
        :param fps: int: target frame per second of animation
        :return: None
        """
        self.fps = fps
        self.game = Game(matrix=matrix) if game is None else game
        self.clock = pygame.time.Clock()
        self.width = width
        self.height = height
        self.margin = margin
        self.block_size = (self.width - (self.game.get_matrix().shape[0] + 1) * margin) // self.game.get_matrix().shape[
            0]
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('2048')

    def run(self) -> None:
        """
        main logic for running the game for playing the game manually
        :return: None
        """
        self.update_ui()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)
                elif event.type == pygame.KEYDOWN and event.key in KEY_MAP:
                    key = KEY_MAP[event.key]
                    if key == K_q:
                        pygame.quit()
                        print("quit 2048")
                        sys.exit(0)
                    elif key == K_r:
                        self.game.move(action=K_r, inplace=True)  # restart
                    elif key in (UP, DOWN, LEFT, RIGHT):
                        self._do_move(key)
            self.update_ui()
            self.clock.tick(self.fps)

    def _do_move(self, direction: str) -> None:
        """
        Apply a directional move and animate the tiles sliding (then the new tile popping in).
        :param direction: one of UP/DOWN/LEFT/RIGHT
        :return: None
        """
        before = self.game.get_matrix().copy()
        afterstate, slides = self._plan_move(before, direction)
        _, _, changed = self.game.move(action=direction, inplace=True)
        if not changed:
            return
        final = self.game.get_matrix()
        self._animate_move(slides)
        spawn = self._find_spawn(afterstate, final)
        if spawn is not None:
            self._animate_spawn(spawn)

    def set_game(self, game: Game) -> None:
        """
        Setter for game object
        :param game: Game: game object containing all game state and logic
        :return: None
        """
        self.game = game

    # ------------------------------------------------------------------ #
    # Move planning — per-tile slide paths (mirrors the web engine's       #
    # collapseLineWithPaths, so the animation matches the actual merges).  #
    # ------------------------------------------------------------------ #
    def _lines(self, direction: str):
        """The four collapse lines as lists of (row, col); index 0 = the edge tiles slide toward."""
        n = self.game.get_matrix().shape[0]
        lines = []
        for k in range(n):
            if direction == LEFT:
                line = [(k, c) for c in range(n)]
            elif direction == RIGHT:
                line = [(k, c) for c in range(n - 1, -1, -1)]
            elif direction == UP:
                line = [(r, k) for r in range(n)]
            elif direction == DOWN:
                line = [(r, k) for r in range(n - 1, -1, -1)]
            else:
                raise ValueError(f"Invalid direction: {direction}")
            lines.append(line)
        return lines

    @staticmethod
    def _collapse_line_paths(vals):
        """
        Greedy left-pairing collapse of one line of tile values (0 = empty), tracking
        where each tile goes. Returns (result_values, moves) where each move is
        (from_pos, to_pos, merged) in line-order positions.
        """
        tiles = [(i, v) for i, v in enumerate(vals) if v != 0]
        result = [0] * len(vals)
        moves = []
        target = 0
        k = 0
        while k < len(tiles):
            if k + 1 < len(tiles) and tiles[k][1] == tiles[k + 1][1]:
                result[target] = tiles[k][1] * 2
                moves.append((tiles[k][0], target, True))
                moves.append((tiles[k + 1][0], target, True))
                k += 2
            else:
                result[target] = tiles[k][1]
                moves.append((tiles[k][0], target, False))
                k += 1
            target += 1
        return result, moves

    def _plan_move(self, matrix, direction):
        """
        Compute the afterstate (before spawn) and the list of tile slides for a move.
        :return: (afterstate matrix, slides) where each slide is
                 (from_row, from_col, to_row, to_col, value, merged)
        """
        after = matrix.copy()
        slides = []
        for line in self._lines(direction):
            vals = [int(matrix[r, c]) for (r, c) in line]
            result, moves = self._collapse_line_paths(vals)
            for p, (r, c) in enumerate(line):
                after[r, c] = result[p]
            for (fp, tp, merged) in moves:
                fr, fc = line[fp]
                tr, tc = line[tp]
                slides.append((fr, fc, tr, tc, vals[fp], merged))
        return after, slides

    @staticmethod
    def _find_spawn(afterstate, final):
        """Locate the cell that went from empty (afterstate) to filled (final) — the spawned tile."""
        n = afterstate.shape[0]
        for r in range(n):
            for c in range(n):
                if afterstate[r, c] == 0 and final[r, c] != 0:
                    return (r, c, int(final[r, c]))
        return None

    # ------------------------------------------------------------------ #
    # Animation                                                           #
    # ------------------------------------------------------------------ #
    def _cell_xy(self, row_i: int, col_i: int):
        """Pixel top-left of a cell (same layout as `_draw_grid`)."""
        x = col_i * self.block_size + (col_i + 1) * self.margin
        y = row_i * self.block_size + (row_i + 1) * self.margin
        return x, y

    def _draw_tile(self, x: float, y: float, val: int, size: float = None) -> None:
        """Draw a single tile (optionally scaled, centered in its cell) at pixel (x, y)."""
        if size is None:
            size = self.block_size
        off = (self.block_size - size) / 2
        rect = pygame.Rect(int(x + off), int(y + off), int(size), int(size))
        pygame.draw.rect(self.screen, get_bg_color(int(val)), rect)
        if int(val) != 0 and size > self.block_size * 0.55:
            font = pygame.font.Font(None, 64)
            text = font.render(str(int(val)), True, (255, 255, 255))
            text_position = text.get_rect()
            text_position.center = rect.center
            self.screen.blit(text, text_position)

    def _draw_empty_grid(self, skip=None) -> None:
        """Draw the board background: every cell as an empty slot (optionally skipping one)."""
        n = self.game.get_matrix().shape[0]
        for r in range(n):
            for c in range(n):
                if skip is not None and (r, c) == skip:
                    continue
                x, y = self._cell_xy(r, c)
                pygame.draw.rect(self.screen, get_bg_color(0),
                                 pygame.Rect(x, y, self.block_size, self.block_size))

    def _animate_move(self, slides) -> None:
        """Slide every tile from its source cell to its destination over ~90ms (ease-out)."""
        steps = 6
        for step in range(1, steps + 1):
            t = step / steps
            e = 1 - (1 - t) * (1 - t)  # ease-out
            self.screen.fill(BG_COLOR)
            self._draw_empty_grid()
            for (fr, fc, tr, tc, val, _merged) in slides:
                if val == 0:
                    continue
                x0, y0 = self._cell_xy(fr, fc)
                x1, y1 = self._cell_xy(tr, tc)
                self._draw_tile(x0 + (x1 - x0) * e, y0 + (y1 - y0) * e, val)
            self._update_score()
            self._update_msg()
            pygame.display.flip()
            pygame.time.delay(15)

    def _animate_spawn(self, spawn) -> None:
        """Pop the newly spawned tile in from ~20% to full size over ~48ms."""
        r, c, val = spawn
        steps = 4
        x, y = self._cell_xy(r, c)
        for step in range(1, steps + 1):
            s = step / steps
            self.screen.fill(BG_COLOR)
            self._draw_final_grid(skip=(r, c))
            self._draw_tile(x, y, val, size=self.block_size * (0.2 + 0.8 * s))
            self._update_score()
            self._update_msg()
            pygame.display.flip()
            pygame.time.delay(12)

    def _draw_final_grid(self, skip=None) -> None:
        """Draw the current (post-move) board, optionally leaving one cell empty."""
        font = pygame.font.Font(None, 64)
        matrix = self.game.get_matrix()
        num_row, num_col = matrix.shape
        for row_i in range(num_row):
            for col_i in range(num_col):
                x, y = self._cell_xy(row_i, col_i)
                rect = pygame.Rect(x, y, self.block_size, self.block_size)
                if skip is not None and (row_i, col_i) == skip:
                    pygame.draw.rect(self.screen, get_bg_color(0), rect)
                    continue
                cell_val = int(matrix[row_i, col_i])
                pygame.draw.rect(self.screen, get_bg_color(cell_val), rect)
                if cell_val != 0:
                    text = font.render(str(cell_val), True, (255, 255, 255))
                    text_position = text.get_rect()
                    text_position.center = rect.center
                    self.screen.blit(text, text_position)

    def _update_score(self) -> None:
        """
        update score on UI
        :return: None
        """
        font = pygame.font.Font(None, 64)
        text = font.render(
            'Score: ' + str(self.game.get_score()), 30, (255, 255, 255))
        self.screen.blit(text, (50, 820))

    def _update_msg(self) -> None:
        """
        update messages displayed on UI
        :return: None
        """
        font = pygame.font.Font(None, 32)
        text = font.render('Game ends, press r to restart' if self.game.get_is_done(
        ) else "Click 'q' to quit the game", True, (255, 255, 255))
        self.screen.blit(text, (50, 870))

    def _draw_grid(self) -> None:
        """
        draw the game board grid on UI
        :return: None
        """
        self._draw_final_grid()

    def update_ui(self) -> None:
        """
        Update UI altogether including updating score, messages displayed and grid
        :return: None
        """
        self.clock.tick(self.fps)
        self.screen.fill(BG_COLOR)
        self._draw_grid()
        self._update_score()
        self._update_msg()
        pygame.display.flip()
