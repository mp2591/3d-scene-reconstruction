"""Hiera-T + HieraNeRF.

* ``encoder.py``: the Hiera-T image encoder (registered as ``hiera``), whose
  first three stages are fused into one feature map per image.
* ``bridge.py``: turns a 3D point into one embedding per source photo.
* ``attention.py``: the attention blocks, including the Hiera-style ray step.
* ``decoder.py``: the four-step attention decoder (``hiera_ray_attention``).
* ``model.py``: the top-level model (``hiera_nerf``).

See ``docs/hiera_nerf.md``.
"""
