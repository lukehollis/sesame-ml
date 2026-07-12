"""Vectorized sim-to-real randomization for the Sesame MJX model."""

from __future__ import annotations

import functools
from collections.abc import Callable

try:
    import jax
    import jax.numpy as jp
    import mujoco
    from mujoco import mjx
except ImportError as exc:  # pragma: no cover - guarded by the lazy package boundary.
    raise ImportError(
        "sesame_ml.playground.randomize requires JAX and MuJoCo Playground dependencies."
    ) from exc


def domain_randomize(
    model: mjx.Model,
    rng: jax.Array,
    *,
    base_body_id: int,
    floor_geom_id: int,
) -> tuple[mjx.Model, mjx.Model]:
    """Create a batched model with mass, COM, friction, and servo variation.

    ``rng`` is supplied in batched form by Brax PPO.  The returned ``in_axes``
    tree tells Playground which model leaves carry an environment dimension.
    """

    @jax.vmap
    def randomize_one(key: jax.Array) -> tuple[jax.Array, ...]:
        keys = jax.random.split(key, 7)

        link_scale = jax.random.uniform(keys[0], (model.nbody,), minval=0.85, maxval=1.15)
        link_scale = link_scale.at[0].set(1.0)  # World mass remains zero.
        body_mass = model.body_mass * link_scale
        body_inertia = model.body_inertia * link_scale[:, None]

        payload = jax.random.uniform(keys[1], minval=-0.025, maxval=0.080)
        old_base_mass = jp.maximum(body_mass[base_body_id], 1e-6)
        new_base_mass = jp.maximum(0.05, old_base_mass + payload)
        body_mass = body_mass.at[base_body_id].set(new_base_mass)
        body_inertia = body_inertia.at[base_body_id].multiply(new_base_mass / old_base_mass)

        com_offset = jax.random.uniform(keys[2], (3,), minval=-0.004, maxval=0.004)
        body_ipos = model.body_ipos.at[base_body_id].add(com_offset)

        friction_scale = jax.random.uniform(keys[3], minval=0.65, maxval=1.30)
        geom_friction = model.geom_friction.at[:, 0].multiply(friction_scale)
        floor_friction = jax.random.uniform(keys[4], minval=0.65, maxval=1.25)
        geom_friction = geom_friction.at[floor_geom_id, 0].set(floor_friction)

        servo_gain_scale = jax.random.uniform(keys[5], (model.nu,), minval=0.72, maxval=1.20)
        actuator_gainprm = model.actuator_gainprm.at[:, 0].multiply(servo_gain_scale)
        actuator_biasprm = model.actuator_biasprm.at[:, 1].multiply(servo_gain_scale)
        strength_scale = jax.random.uniform(keys[6], (model.nu, 1), minval=0.75, maxval=1.10)
        actuator_forcerange = model.actuator_forcerange * strength_scale

        return (
            body_mass,
            body_inertia,
            body_ipos,
            geom_friction,
            actuator_gainprm,
            actuator_biasprm,
            actuator_forcerange,
        )

    randomized = randomize_one(rng)
    fields = (
        "body_mass",
        "body_inertia",
        "body_ipos",
        "geom_friction",
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_forcerange",
    )
    replacements = dict(zip(fields, randomized, strict=True))
    in_axes = jax.tree_util.tree_map(lambda _: None, model)
    in_axes = in_axes.tree_replace({name: 0 for name in fields})
    return model.tree_replace(replacements), in_axes


def make_domain_randomizer(
    model: mujoco.MjModel,
) -> Callable[[mjx.Model, jax.Array], tuple[mjx.Model, mjx.Model]]:
    """Bind name-derived CPU model IDs into the JIT-safe randomizer."""

    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if base_body_id < 0 or floor_geom_id < 0:
        raise ValueError("Sesame model must contain the base body and floor geom")
    return functools.partial(
        domain_randomize,
        base_body_id=base_body_id,
        floor_geom_id=floor_geom_id,
    )
