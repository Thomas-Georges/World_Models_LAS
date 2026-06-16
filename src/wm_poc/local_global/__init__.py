"""Local/global world-model planning track.

The global model (DINO-WM, frozen) provides trusted forward rollouts in DINO
patch-latent space; a small differentiable local surrogate provides action
gradients; hybrid planners combine global search with local refinement.
See LOCAL_GLOBAL_DINO_WM_IMPLEMENTATION_SPEC.md at the repository root.
"""
