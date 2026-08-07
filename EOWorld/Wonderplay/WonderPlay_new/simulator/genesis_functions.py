"""Scene-specific force functions for Genesis simulation (venice_1 only)."""
import numpy as np


def force_venice_1(obj_entity, obj_id, sid, all_entities):
    time_lapse = 1000
    magnitude_0 = 65
    force = np.array([0, 0, 0])

    if obj_id == 0:
        if sid <= time_lapse:
            force_direction = np.array([1, -0.1, 0])
            force = magnitude_0 * force_direction / np.linalg.norm(force_direction)
        else:
            force = np.array([0, 0, 0])

    force = force.reshape(1, 3)
    obj_idx = obj_entity.idx
    obj_entity.solver.apply_links_external_force(
        force=force,
        links_idx=[obj_idx],
    )
    return None
