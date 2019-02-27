# `settings.npy`

Contains individual (one-person) complete settings containing a cup, a fork, and a knife each.
The settings are rotated so that they are roughly aligned according to a person who would try to eat from the seat, with the plate at `(0, 0)`.

Data format:

- `cup_x`, `cup_y`, `fork_x`, `fork_y`, `knife_x`, `knife_y`


# `regular_items.npy`

Contains the items from `settings.npy` (i.e. the same "normalization" occurred), but each row is one item., compared to one row per setting.
Thus, this file contains three times as many entries as `settings.npy`.

Data format:
- `x`, `y`, `label`

With labels:
- 0: cup
- 1: fork
- 2: knife
