# Data Manifest

This repository does not store datasets.

## Track 1 - DreamerV3 / R2-Dreamer

R2-Dreamer/DreamerV3 online RL experiments collect data through environment interaction.

Initial environments:
- DMC Proprio
- DMC Vision
- Optional: Meta-World

No static dataset is required for the first DMC runs.

## Track 2 - DINO-WM / local-global planning

DINO-WM-style experiments use offline trajectory datasets.

Candidate datasets:
- PointMaze
- PushT
- Wall
- Optional later: Rope
- Optional later: Granular

Expected Drive location:

```text
/content/drive/MyDrive/wm_poc/data/dino_wm/
```

Expected subfolders, once downloaded:

```text
point_maze/
pusht_noise/
wall_single/
rope/
granular/
```

## Dataset policy

- Do not commit datasets to GitHub.
- Store datasets under Google Drive or another persistent artifact store.
- Record dataset source URL, download date, checksum if available, and local path.
- Prefer official dataset links from the upstream repositories.
