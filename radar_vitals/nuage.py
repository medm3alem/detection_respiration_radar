"""Nuage de points de la demo in-cabin : conversions de repere et zones.

Conventions reprises de parseDataFromDevice.m (visualiseur MATLAB TI) :
    repere capteur : x = r.cos(el).sin(az), y = r.cos(el).cos(az), z = r.sin(el)
    repere monde   : Rot_TW * capteur + sensorPosition   (cfg_parser.sensor_to_world)
"""
from __future__ import annotations

import numpy as np

from .cfg_parser import RadarConfig

ZONE_HORS = -1


def spherical_to_sensor(points: np.ndarray, flip_azimuth: bool = False, flip_elevation: bool = False) -> np.ndarray:
    """points [N, >=3] (distance, azimut, elevation) -> xyz [N, 3] dans le repere capteur."""
    if points is None or len(points) == 0:
        return np.zeros((0, 3))
    r, az, el = points[:, 0], points[:, 1], points[:, 2]
    if flip_azimuth:
        az = -az
    if flip_elevation:
        el = -el
    return np.column_stack([r * np.cos(el) * np.sin(az), r * np.cos(el) * np.cos(az), r * np.sin(el)])


def sensor_to_spherical(xyz: np.ndarray) -> np.ndarray:
    """xyz capteur [N, 3] -> (distance, azimut, elevation) [N, 3]."""
    r = np.linalg.norm(xyz, axis=1)
    az = np.arctan2(xyz[:, 0], xyz[:, 1])
    el = np.arcsin(np.clip(xyz[:, 2] / np.maximum(r, 1e-9), -1, 1))
    return np.column_stack([r, az, el])


def zone_of_points(cfg: RadarConfig, xyz_world: np.ndarray) -> np.ndarray:
    """Indice de zone (siege) de chaque point, ZONE_HORS s'il n'est dans aucun cuboide."""
    zones = np.full(len(xyz_world), ZONE_HORS, dtype=int)
    if len(xyz_world) == 0:
        return zones
    x, y, z = xyz_world[:, 0], xyz_world[:, 1], xyz_world[:, 2]
    for zone in sorted(cfg.cuboids):
        for x1, x2, y1, y2, z1, z2 in cfg.cuboids[zone]:
            inside = (x >= x1) & (x <= x2) & (y >= y1) & (y <= y2) & (z >= z1) & (z <= z2) & (zones == ZONE_HORS)
            zones[inside] = zone
    return zones


def cuboid_corners(box: tuple[float, ...]) -> np.ndarray:
    x1, x2, y1, y2, z1, z2 = box
    return np.array([[x, y, z] for x in (x1, x2) for y in (y1, y2) for z in (z1, z2)])


# aretes d'un cuboide (indices dans cuboid_corners)
CUBOID_EDGES = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]


def count_per_zone(zones: np.ndarray, num_zones: int) -> np.ndarray:
    return np.array([(zones == z).sum() for z in range(num_zones)])
