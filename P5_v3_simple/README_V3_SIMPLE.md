# CMPM 146 Programming #5 - Simplified V3

This version keeps the project close to the provided course template. It implements the required genetic algorithm features without optional systems such as FI-2POP, novelty search, behavior archives, checkpoint portfolios, or automatic restarts.

## Required features implemented

### Selection

`generate_successors()` uses two selection strategies:

1. **Elitist selection** keeps the best 10% of the current population.
2. **Tournament selection** randomly samples four individuals and selects the best one as a parent.

Parents create children until the next population has the same size as the current population.

### Grid encoding

- **Crossover:** column-based single-point crossover.
- **Mutation rate:** 20% per child.
- **Mutation operations:** change one tile, toggle a short hole, or add one complete pipe.
- **Simple repair:** restores the start and goal, limits holes to four tiles, and removes unsupported enemies.
- **Initialization:** starts with flat ground and adds a small number of holes, pipes, platforms, enemies, and rewards.

### Design Element encoding

- Keeps the provided variable-point crossover idea.
- Fixes crossover when one parent's genome is empty.
- Uses a 20% mutation rate.
- Mutation can change, add, or delete one design element.
- Uses conservative ranges for holes, platforms, stairs, and pipes.
- Adds simple penalties for excessive holes, enemies, stairs, and genome length.

### Fitness

Both representations use the supplied `metrics.py` measurements. Solvability has the largest weight. Smaller weights encourage reachable space, meaningful jumps, decoration, and less perfectly linear terrain.

## Switching encodings

In `ga.py`, use:

```python
Individual = Individual_Grid
```

For the Design Element version, change it to:

```python
Individual = Individual_DE
```

## Running

From the directory containing `ga.py`, `metrics.py`, and `pathfinding.py`:

```powershell
py .\ga.py
```

The program runs until `Ctrl-C` is pressed. The best level from the current generation is written to:

```text
levels/last.txt
```

When the program exits, the ten best individuals in the final population are saved as timestamped text files.

## Files changed

- `ga.py`: modified for the assignment.
- `metrics.py`: unchanged from the provided template.
- `pathfinding.py`: unchanged from the provided template.

## Removed from the earlier V3

The earlier V3 contained several features outside the assignment's normal scope. This simplified version removes:

- regional entropy calculations
- behavioral novelty bonuses
- genotype/behavior diversity tracking
- global-best archives and portfolios
- checkpoint files
- stagnation detection and automatic restarts
- CSV and JSON experiment logging
- large structural repair passes

These systems were not required by the rubric and made the group code harder to explain. The simplified version focuses directly on selection, crossover, mutation, fitness, and the two required encodings.
