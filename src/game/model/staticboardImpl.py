import random
from typing import Union, Tuple, List

import numpy as np
import pandas as pd
import torch
from numba import njit
from torch import Tensor

from src.game.model.staticboard import StaticBoard
from src.game.utils import UP, DOWN, LEFT, RIGHT, SEED


class NumpyStaticBoard(StaticBoard):
    seed = SEED

    @staticmethod
    def set_random_seed(seed: int = 2048):
        random.seed(seed)
        np.random.seed(seed)

    @staticmethod
    def get_pd_df(matrix: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(data=matrix).astype(int)

    @staticmethod
    @njit
    def get_empty_coordinates(matrix: np.ndarray) -> np.ndarray:
        return np.argwhere(matrix == 0)

    @staticmethod
    @njit
    def has_empty_cell(matrix: np.ndarray) -> bool:
        return matrix.min() == 0

    @staticmethod
    def has_won(matrix: np.ndarray, goal: int = 2048) -> bool:
        return goal in matrix

    @staticmethod
    @njit
    def get_max_val(matrix: np.ndarray) -> int:
        return matrix.max()

    @staticmethod
    def get_random_empty_cell_coordinate(matrix: np.ndarray) -> Union[np.array, None]:
        empty_cell_coordinates = np.argwhere(matrix == 0)
        return (
            None
            if len(empty_cell_coordinates) == 0
            else empty_cell_coordinates[
                np.random.randint(0, len(empty_cell_coordinates))
            ]
        )

    @staticmethod
    def get_empty_matrix(width: int = 4, height: int = 4) -> np.ndarray:
        return np.zeros((width, height)).astype("int64")

    @staticmethod
    def set_random_cell(
        matrix: np.ndarray, inplace: bool = True
    ) -> Tuple[np.ndarray, bool]:
        empty_cell_coordinates = np.argwhere(matrix == 0)
        empty_cell = (
            None
            if len(empty_cell_coordinates) == 0
            else empty_cell_coordinates[
                np.random.randint(0, len(empty_cell_coordinates))
            ]
        )
        if not inplace:
            matrix = matrix.copy()
        if empty_cell is None:
            return matrix, False
        else:
            is_4 = random.random() < 0.1
            matrix[empty_cell[0], empty_cell[1]] = 4 if is_4 else 2
            return matrix, True

    @staticmethod
    def get_init_matrix(width: int = 4, height: int = 4) -> np.ndarray:
        mesh_grid = np.meshgrid(np.arange(width), np.arange(height))
        coordinates = np.stack(mesh_grid, axis=2).reshape(width * height, 2)
        rand_coordinates = coordinates[
            np.random.choice(len(coordinates), 2, replace=False)
        ]
        matrix = np.zeros((width, height))
        for i in range(2):
            matrix[rand_coordinates[i][0], rand_coordinates[i][1]] = 2
        return matrix

    @staticmethod
    def compute_is_done(matrix: np.ndarray) -> bool:
        h, w = matrix.shape
        for row_i in range(h):
            row = matrix[row_i, :]
            for col_i in range(len(row)):
                if matrix[row_i, col_i] == 0:
                    return False
                neighbor_coordinates = NumpyStaticBoard.get_neighbors_coordinates(
                    matrix, row_i, col_i
                )
                for neighbor in neighbor_coordinates:
                    if matrix[neighbor[0], neighbor[1]] == matrix[row_i, col_i]:
                        return False
        return True

    @staticmethod
    @njit
    def collapse_array(arr: np.ndarray, reverse: bool = False) -> np.array:
        arr_len = len(arr)
        changed = False
        score = 0
        has_merged_arr = np.zeros_like(arr)
        for i in range(
            1 if reverse else arr_len - 2,
            arr_len if reverse else -1,
            1 if reverse else -1,
        ):
            curr_i = i
            for next_i in range(
                i + (-1 if reverse else 1),
                -1 if reverse else arr_len,
                -1 if reverse else 1,
            ):
                if arr[next_i] == 0 and arr[curr_i] != 0:
                    arr[next_i] = arr[curr_i]
                    arr[curr_i] = 0
                    curr_i = next_i
                    changed = True
                elif (
                    arr[curr_i] == arr[next_i]
                    and not has_merged_arr[next_i]
                    and arr[curr_i] != 0
                ):
                    # merge case
                    arr[next_i] *= 2
                    arr[curr_i] = 0
                    has_merged_arr[next_i] = 1
                    score += int(arr[next_i])
                    changed = True
                    break
                else:
                    break
        return arr, score, changed

    @staticmethod
    def move(
        matrix: np.ndarray,
        direction: Union[UP, DOWN, LEFT, RIGHT],
        inplace: bool = True,
    ) -> Tuple[np.ndarray, int, bool]:
        score = 0
        changed = False
        if not inplace:
            matrix = matrix.copy()
        for i in range(len(matrix)):
            if direction == UP:
                arr, score_, changed_ = NumpyStaticBoard.collapse_array(
                    matrix[:, i], reverse=True
                )
            elif direction == DOWN:
                arr, score_, changed_ = NumpyStaticBoard.collapse_array(
                    matrix[:, i], reverse=False
                )
            elif direction == LEFT:
                arr, score_, changed_ = NumpyStaticBoard.collapse_array(
                    matrix[i, :], reverse=True
                )
            elif direction == RIGHT:
                arr, score_, changed_ = NumpyStaticBoard.collapse_array(
                    matrix[i, :], reverse=False
                )
            else:
                raise ValueError(f"Invalid direction: {direction}")
            changed = max(changed, changed_)
            score += score_
        return matrix, score, changed


class TorchStaticBoard(object):
    seed = SEED

    @staticmethod
    def get_pd_df(matrix: Tensor) -> pd.DataFrame:
        return pd.DataFrame(data=matrix).astype(int)

    @staticmethod
    def get_empty_coordinates(matrix: Tensor) -> Tensor:
        return torch.nonzero(matrix == 0)

    @staticmethod
    def has_empty_cell(matrix: Tensor) -> bool:
        return len(TorchStaticBoard.get_empty_coordinates(matrix)) != 0

    @staticmethod
    def has_won(matrix: Tensor, goal: int = 2048):
        return goal in matrix

    @staticmethod
    def get_max_val(matrix: Tensor):
        return int(torch.max(matrix))

    @staticmethod
    def get_random_empty_cell_coordinate(matrix: Tensor) -> Union[Tensor, None]:
        empty_cell_coordinates = TorchStaticBoard.get_empty_coordinates(matrix)
        return (
            None
            if len(empty_cell_coordinates) == 0
            else empty_cell_coordinates[
                np.random.randint(0, len(empty_cell_coordinates))
            ]
        )

    @staticmethod
    def get_empty_matrix(width: int = 4, height: int = 4):
        return torch.zeros((width, height)).int()

    @staticmethod
    def get_init_matrix(width: int = 4, height: int = 4):
        matrix = TorchStaticBoard.get_empty_matrix(width=width, height=height)
        for _ in range(2):
            TorchStaticBoard.set_random_cell(matrix, inplace=True)
        return matrix

    @staticmethod
    def set_random_cell(matrix: Tensor, inplace: bool = True) -> Tuple[Tensor, bool]:
        empty_cell = TorchStaticBoard.get_random_empty_cell_coordinate(matrix)
        if not inplace:
            matrix = matrix.clone()
        if empty_cell is None:
            return matrix, False
        else:
            is_4 = random.random() < 0.1
            matrix[empty_cell[0], empty_cell[1]] = 4 if is_4 else 2
            return matrix, True

    @staticmethod
    def get_neighbors_coordinates(matrix: Tensor, row_i: int, col_i: int) -> List:
        result = []
        h, w = matrix.shape
        if row_i > 0:
            result.append((row_i - 1, col_i))
        if row_i < h - 1:
            result.append((row_i + 1, col_i))
        if col_i > 0:
            result.append((row_i, col_i - 1))
        if col_i < w - 1:
            result.append((row_i, col_i + 1))
        return result

    @staticmethod
    def compute_is_done(matrix: Tensor):
        h, w = matrix.shape
        for row_i in range(h):
            row = matrix[row_i, :]
            for col_i in range(len(row)):
                neighbor_coordinates = TorchStaticBoard.get_neighbors_coordinates(
                    matrix, row_i, col_i
                )
                for neighbor in neighbor_coordinates:
                    if matrix[neighbor[0], neighbor[1]] == matrix[row_i, col_i]:
                        return False
                    if matrix[row_i, col_i] == 0:
                        return False
        return True

    @staticmethod
    def move(
        matrix: Tensor, direction: Union[UP, DOWN, LEFT, RIGHT], inplace: bool = True
    ) -> Tuple[Tensor, bool]:
        score = 0
        changed = False
        if not inplace:
            matrix = matrix.clone()
        for i in range(len(matrix)):
            if direction == UP:
                arr, score_, changed_ = TorchStaticBoard.collapse_array(
                    matrix[:, i], reverse=True
                )
            elif direction == DOWN:
                arr, score_, changed_ = TorchStaticBoard.collapse_array(
                    matrix[:, i], reverse=False
                )
            elif direction == LEFT:
                arr, score_, changed_ = TorchStaticBoard.collapse_array(
                    matrix[i, :], reverse=True
                )
            elif direction == RIGHT:
                arr, score_, changed_ = TorchStaticBoard.collapse_array(
                    matrix[i, :], reverse=False
                )
            else:
                raise ValueError(f"Invalid direction: {direction}")
            changed = max(changed, changed_)
            score += score_
        return matrix, score, changed

    @staticmethod
    def collapse_array(arr: Tensor, reverse=False):
        """
        collapse an array in left-to-right direction in 2048-way inplace, this is either a row or an column
        :param array: Tensor Array
        :return:
        """
        arr_len = len(arr)
        changed = False
        score = 0
        has_merged_arr = torch.zeros_like(arr)
        for i in range(
            1 if reverse else arr_len - 2,
            arr_len if reverse else -1,
            1 if reverse else -1,
        ):
            curr_i = i
            for next_i in range(
                i + (-1 if reverse else 1),
                -1 if reverse else arr_len,
                -1 if reverse else 1,
            ):
                if arr[next_i] == 0 and arr[curr_i] != 0:
                    arr[next_i] = arr[curr_i]
                    arr[curr_i] = 0
                    curr_i = next_i
                    changed = True
                elif (
                    arr[curr_i] == arr[next_i]
                    and not has_merged_arr[next_i]
                    and arr[curr_i] != 0
                ):
                    # merge case
                    arr[next_i] *= 2
                    arr[curr_i] = 0
                    has_merged_arr[next_i] = 1
                    score += int(arr[next_i])
                    changed = True
                    break
                else:
                    break
        return arr, score, changed
