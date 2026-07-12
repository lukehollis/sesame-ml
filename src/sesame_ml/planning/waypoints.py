"""Online 2-D route planning for navigation goals produced by vision or language."""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AxisAlignedObstacle:
    center_xy: tuple[float, float]
    half_size_xy: tuple[float, float]

    def contains(self, point: np.ndarray, inflation: float = 0.0) -> bool:
        center = np.asarray(self.center_xy)
        half_size = np.asarray(self.half_size_xy) + inflation
        return bool(np.all(np.abs(point - center) <= half_size))


class OccupancyGridPlanner:
    """A* route planner with footprint inflation and line-of-sight path smoothing."""

    _NEIGHBORS = tuple(
        (dx, dy, float(np.hypot(dx, dy)))
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        if dx or dy
    )

    def __init__(self, resolution_m: float = 0.05, robot_radius_m: float = 0.09) -> None:
        if resolution_m <= 0 or robot_radius_m < 0:
            raise ValueError("resolution must be positive and radius non-negative")
        self.resolution = resolution_m
        self.robot_radius = robot_radius_m

    def _blocked(
        self, point: np.ndarray, obstacles: tuple[AxisAlignedObstacle, ...]
    ) -> bool:
        return any(obstacle.contains(point, self.robot_radius) for obstacle in obstacles)

    def _line_clear(
        self,
        start: np.ndarray,
        end: np.ndarray,
        obstacles: tuple[AxisAlignedObstacle, ...],
    ) -> bool:
        distance = float(np.linalg.norm(end - start))
        steps = max(2, int(np.ceil(distance / (self.resolution * 0.5))))
        return not any(
            self._blocked(start + alpha * (end - start), obstacles)
            for alpha in np.linspace(0, 1, steps)
        )

    def plan(
        self,
        start_xy: np.ndarray | tuple[float, float],
        goal_xy: np.ndarray | tuple[float, float],
        obstacles: list[AxisAlignedObstacle] | tuple[AxisAlignedObstacle, ...],
    ) -> np.ndarray:
        start = np.asarray(start_xy, dtype=np.float64)
        goal = np.asarray(goal_xy, dtype=np.float64)
        obstacle_tuple = tuple(obstacles)
        if start.shape != (2,) or goal.shape != (2,):
            raise ValueError("start and goal must be two-dimensional")
        if self._blocked(start, obstacle_tuple) or self._blocked(goal, obstacle_tuple):
            raise ValueError("start or goal lies inside an inflated obstacle")

        margin = max(0.35, self.robot_radius * 2)
        points = [start, goal]
        for obstacle in obstacle_tuple:
            center, half = np.asarray(obstacle.center_xy), np.asarray(obstacle.half_size_xy)
            points.extend([center - half - margin, center + half + margin])
        low = np.min(points, axis=0) - margin
        high = np.max(points, axis=0) + margin

        def to_cell(point: np.ndarray) -> tuple[int, int]:
            cell = np.rint((point - low) / self.resolution).astype(int)
            return int(cell[0]), int(cell[1])

        def to_point(cell: tuple[int, int]) -> np.ndarray:
            return low + self.resolution * np.asarray(cell)

        start_cell, goal_cell = to_cell(start), to_cell(goal)
        max_cell = np.ceil((high - low) / self.resolution).astype(int)
        queue: list[tuple[float, float, tuple[int, int]]] = [(0.0, 0.0, start_cell)]
        parent: dict[tuple[int, int], tuple[int, int]] = {}
        best_cost = {start_cell: 0.0}
        found = False
        while queue:
            _, cost, cell = heapq.heappop(queue)
            if cell == goal_cell:
                found = True
                break
            if cost > best_cost.get(cell, np.inf):
                continue
            for dx, dy, step_cost in self._NEIGHBORS:
                neighbor = cell[0] + dx, cell[1] + dy
                if not (0 <= neighbor[0] <= max_cell[0] and 0 <= neighbor[1] <= max_cell[1]):
                    continue
                if self._blocked(to_point(neighbor), obstacle_tuple):
                    continue
                candidate_cost = cost + step_cost
                if candidate_cost >= best_cost.get(neighbor, np.inf):
                    continue
                best_cost[neighbor] = candidate_cost
                parent[neighbor] = cell
                heuristic = float(np.linalg.norm(np.asarray(neighbor) - np.asarray(goal_cell)))
                heapq.heappush(queue, (candidate_cost + heuristic, candidate_cost, neighbor))
        if not found:
            raise RuntimeError("No collision-free route exists within the planning bounds")

        cells = [goal_cell]
        while cells[-1] != start_cell:
            cells.append(parent[cells[-1]])
        raw = [start, *[to_point(cell) for cell in reversed(cells[1:-1])], goal]
        smoothed = [raw[0]]
        cursor = 0
        while cursor < len(raw) - 1:
            target = len(raw) - 1
            while target > cursor + 1 and not self._line_clear(
                raw[cursor], raw[target], obstacle_tuple
            ):
                target -= 1
            smoothed.append(raw[target])
            cursor = target
        return np.asarray(smoothed, dtype=np.float32)
