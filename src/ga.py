"""Genetic algorithm for evolving Super Mario Bros. levels.

The module supports two genome representations:

* ``Individual_Grid`` stores the complete tile map.
* ``Individual_DE`` stores a variable-length list of design elements.

Both representations use the supplied course metrics, elitist preservation,
tournament selection, crossover, and mutation.
"""

import copy
import heapq
import math
import multiprocessing.pool as mpool
import os
import random
import time

# Restrict numerical libraries to one thread per worker. This prevents
# nested parallelism from exhausting system memory during multiprocessing.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import metrics

# ---------------------------------------------------------------------------
# Level Constants
# ---------------------------------------------------------------------------

width = 200
height = 16

# Tile symbols defined by the supplied Mario level format.
options = [
    "-",  # empty space
    "X",  # solid wall
    "?",  # question block with a coin
    "M",  # question block with a mushroom
    "B",  # breakable block
    "o",  # coin
    "|",  # pipe segment
    "T",  # pipe top
    "E",  # enemy
]

# Tiles treated as solid support by structural validation.
SOLID_TILES = {"X", "?", "M", "B", "|", "T", "v", "f", "m"}


# ---------------------------------------------------------------------------
# Shared Utility Functions
# ---------------------------------------------------------------------------

def clip(lo, value, hi):
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def offset_by_upto(value, variance, min=None, max=None):
    # Apply a normally distributed offset and constrain the result to the
    # specified bounds.
    value += random.normalvariate(0, variance ** 0.5)
    if min is not None and value < min:
        value = min
    if max is not None and value > max:
        value = max
    return int(value)


def repair_boundaries(genome):
    """Restore the required start and goal tiles after genetic operations."""

    # The first and last columns are reserved by the level specification.
    for y in range(height):
        genome[y][0] = "-"
        genome[y][-1] = "-"

    genome[15][0] = "X"
    genome[14][0] = "m"

    genome[7][-1] = "v"
    for y in range(8, 14):
        genome[y][-1] = "f"
    genome[14][-1] = "X"
    genome[15][-1] = "X"
    return genome


def repair_grid(genome):
    """Enforce basic structural constraints on a Grid genome.

    The repair step preserves the required boundaries, limits ground gaps,
    and removes unsupported enemies.
    """
    repair_boundaries(genome)

    # Preserve short ground sections near the start and goal.
    for x in range(1, 5):
        genome[15][x] = "X"
    for x in range(width - 5, width - 1):
        genome[15][x] = "X"

    # Limit each continuous ground gap to four tiles.
    x = 1
    while x < width - 1:
        if genome[15][x] != "-":
            x += 1
            continue

        start = x
        while x < width - 1 and genome[15][x] == "-":
            x += 1

        for fill_x in range(start + 4, x):
            genome[15][fill_x] = "X"

    # Remove enemies that are not supported by a solid tile.
    for y in range(height - 1):
        for x in range(1, width - 1):
            if genome[y][x] == "E" and genome[y + 1][x] not in SOLID_TILES:
                genome[y][x] = "-"

    return repair_boundaries(genome)


# ---------------------------------------------------------------------------
# Shared Fitness Function
# ---------------------------------------------------------------------------

def common_fitness(level):
    """Compute the fitness score used by both genome representations.

    Solvability receives the highest weight. Secondary terms reward useful
    jumps, navigable space, and moderate decoration while penalizing excessive
    obstacle density.
    """
    measurements = metrics.metrics(level)

    score = 0.0
    score += 8.0 * measurements["solvability"]
    score += 1.0 * measurements["negativeSpace"]
    score += 0.5 * measurements["pathPercentage"]

    # Cap jump rewards to prevent excessive obstacle placement from
    # increasing fitness without improving level quality.
    jumps = max(0.0, measurements["jumps"])
    meaningful_jumps = max(0.0, measurements["meaningfulJumps"])
    score += 0.6 * min(meaningful_jumps, 8.0)
    score += 0.1 * min(jumps, 12.0)
    if jumps > 15.0:
        score -= 0.25 * (jumps - 15.0)

    # Reward moderate decoration and penalize excessive visual density.
    decoration = measurements["decorationPercentage"]
    score += 4.0 * min(decoration, 0.06)
    if decoration > 0.08:
        score -= 30.0 * (decoration - 0.08)

    # Penalize levels whose usable space becomes excessively occupied.
    empty_percentage = measurements["emptyPercentage"]
    if empty_percentage < 0.82:
        score -= 20.0 * (0.82 - empty_percentage)

    score -= 0.5 * measurements["linearity"]
    return score


# ---------------------------------------------------------------------------
# Grid Representation
# ---------------------------------------------------------------------------

class Individual_Grid(object):
    """Represent a Mario level as a complete two-dimensional tile grid."""

    # Restrict instance attributes to reduce per-individual memory usage.
    __slots__ = ["genome", "_fitness"]

    def __init__(self, genome):
        self.genome = copy.deepcopy(genome)
        self._fitness = None

    def calculate_fitness(self):
        # Cache the score because selection may evaluate the same individual
        # multiple times within a generation.
        self._fitness = common_fitness(self.to_level())
        return self

    def fitness(self):
        if self._fitness is None:
            self.calculate_fitness()
        return self._fitness

    def mutate(self, genome):
        """Apply one Grid mutation with a 10 percent mutation probability.

        The operator modifies a single tile, toggles a short ground gap, or
        adds or removes a complete pipe structure.
        """
        if random.random() >= 0.10:
            return repair_grid(genome)

        choice = random.random()

        if choice < 0.70:
            # Modify one non-boundary tile. Pipe tiles are excluded because
            # pipes are generated as complete structures.
            x = random.randint(5, width - 6)
            y = random.randint(6, height - 2)
            if genome[y][x] == "-":
                genome[y][x] = random.choices(
                    ["X", "B", "?", "M", "o", "E"],
                    weights=[25, 20, 15, 3, 30, 7],
                    k=1,
                )[0]
            elif random.random() < 0.65:
                genome[y][x] = "-"
            else:
                genome[y][x] = random.choice(["X", "B", "?", "o"])

        elif choice < 0.85:
            # Toggle a ground gap with a width of one to three tiles.
            x = random.randint(5, width - 9)
            hole_width = random.randint(1, 3)
            make_hole = all(genome[15][x + dx] != "-" for dx in range(hole_width))
            for dx in range(hole_width):
                genome[15][x + dx] = "-" if make_hole else "X"

        else:
            # Add or remove one complete vertical pipe.
            x = random.randint(5, width - 6)
            has_pipe = any(genome[y][x] in {"T", "|"} for y in range(height))
            for y in range(height):
                if genome[y][x] in {"T", "|"}:
                    genome[y][x] = "-"
            if not has_pipe:
                pipe_height = random.randint(2, 4)
                top_y = height - pipe_height - 1
                genome[top_y][x] = "T"
                for y in range(top_y + 1, height):
                    genome[y][x] = "|"

        return repair_grid(genome)

    def generate_children(self, other):
        """Create two children using column-based single-point crossover."""

        # A vertical crossover point preserves contiguous horizontal sections
        # from each parent.
        cut = random.randint(5, width - 6)

        child_a = copy.deepcopy(self.genome)
        child_b = copy.deepcopy(other.genome)

        for y in range(height):
            child_a[y][cut:width - 1] = other.genome[y][cut:width - 1]
            child_b[y][cut:width - 1] = self.genome[y][cut:width - 1]

        child_a = self.mutate(child_a)
        child_b = self.mutate(child_b)

        return Individual_Grid(child_a), Individual_Grid(child_b)

    def to_level(self):
        return self.genome

    @classmethod
    def empty_individual(cls):
        # Initialize an empty level with a continuous ground row.
        genome = [["-" for _x in range(width)] for _y in range(height)]
        genome[15][:] = ["X"] * width
        repair_boundaries(genome)
        return cls(genome)

    @classmethod
    def random_individual(cls):
        """Create a random Grid individual from a valid flat level."""
        genome = cls.empty_individual().genome

        # Traverse the level from left to right and place structures at
        # probabilistically selected positions.
        x = 5
        while x < width - 6:
            roll = random.random()

            if roll < 0.035:
                hole_width = random.randint(1, 3)
                for dx in range(hole_width):
                    genome[15][x + dx] = "-"
                x += hole_width

            elif roll < 0.060:
                pipe_height = random.randint(2, 4)
                top_y = height - pipe_height - 1
                genome[top_y][x] = "T"
                for y in range(top_y + 1, height):
                    genome[y][x] = "|"
                x += 2

            elif roll < 0.110:
                platform_width = random.randint(2, 6)
                y = random.randint(9, 12)
                tile = random.choice(["X", "B", "?"])
                for dx in range(platform_width):
                    if x + dx < width - 5:
                        genome[y][x + dx] = tile
                x += platform_width

            elif roll < 0.160 and genome[15][x] == "X":
                genome[14][x] = "E"
                x += 1

            elif roll < 0.240:
                y = random.randint(8, 13)
                genome[y][x] = random.choice(["o", "o", "?", "B", "M"])
                x += 1

            else:
                x += 1

        return cls(repair_grid(genome))


# ---------------------------------------------------------------------------
# Design Element Representation
#
# Each genome entry has the form (x, element_type, ...parameters).
# Example: (40, "0_hole", 3) represents a three-tile gap beginning at x = 40.
# ---------------------------------------------------------------------------

def random_design_element():
    """Create one valid design element for the DE encoding."""
    x = random.randint(5, width - 6)
    de_type = random.randint(0, 7)

    if de_type == 0:
        return (x, "0_hole", random.randint(1, 4))
    if de_type == 1:
        return (
            x,
            "1_platform",
            random.randint(2, 8),
            random.randint(2, 7),
            random.choice(["?", "X", "B"]),
        )
    if de_type == 2:
        return (x, "2_enemy")
    if de_type == 3:
        return (x, "3_coin", random.randint(6, 13))
    if de_type == 4:
        return (x, "4_block", random.randint(6, 13), random.choice([True, False]))
    if de_type == 5:
        return (x, "5_qblock", random.randint(6, 13), random.choice([True, False]))
    if de_type == 6:
        return (x, "6_stairs", random.randint(1, 5), random.choice([-1, 1]))
    return (x, "7_pipe", random.randint(2, 4))


class Individual_DE(object):
    """A level represented as a variable-length list of design elements."""

    __slots__ = ["genome", "_fitness", "_level"]

    def __init__(self, genome):
        self.genome = list(genome)

        # Preserve the heap-based genome organization used by the starter
        # implementation.
        heapq.heapify(self.genome)
        self._fitness = None
        self._level = None

    def calculate_fitness(self):
        # Evaluate the rendered level using the shared metric-based score.
        score = common_fitness(self.to_level())

        # Simple DE-specific limits. They discourage extreme genomes without
        # adding a separate algorithm such as FI-2POP.
        hole_count = sum(1 for de in self.genome if de[1] == "0_hole")
        enemy_count = sum(1 for de in self.genome if de[1] == "2_enemy")
        stair_count = sum(1 for de in self.genome if de[1] == "6_stairs")

        if hole_count > 8:
            score -= 0.5 * (hole_count - 8)
        if enemy_count > 15:
            score -= 0.25 * (enemy_count - 15)
        if stair_count > 5:
            score -= 0.5 * (stair_count - 5)
        if len(self.genome) > 60:
            score -= 0.1 * (len(self.genome) - 60)

        self._fitness = score
        return self

    def fitness(self):
        if self._fitness is None:
            self.calculate_fitness()
        return self._fitness

    def _change_element(self, de):
        """Make one small parameter change to an existing design element."""
        x = de[0]
        de_type = de[1]
        choice = random.random()

        if de_type == "0_hole":
            hole_width = de[2]
            if choice < 0.5:
                x = offset_by_upto(x, width / 16, min=5, max=width - 6)
            else:
                hole_width = offset_by_upto(hole_width, 2, min=1, max=4)
            return (x, de_type, hole_width)

        if de_type == "1_platform":
            platform_width, platform_height, material = de[2], de[3], de[4]
            if choice < 0.25:
                x = offset_by_upto(x, width / 16, min=5, max=width - 6)
            elif choice < 0.50:
                platform_width = offset_by_upto(platform_width, 3, min=2, max=8)
            elif choice < 0.75:
                platform_height = offset_by_upto(platform_height, 2, min=2, max=7)
            else:
                material = random.choice(["?", "X", "B"])
            return (x, de_type, platform_width, platform_height, material)

        if de_type == "2_enemy":
            x = offset_by_upto(x, width / 16, min=5, max=width - 6)
            return (x, de_type)

        if de_type == "3_coin":
            y = de[2]
            if choice < 0.5:
                x = offset_by_upto(x, width / 16, min=5, max=width - 6)
            else:
                y = offset_by_upto(y, 3, min=6, max=13)
            return (x, de_type, y)

        if de_type == "4_block":
            y, breakable = de[2], de[3]
            if choice < 0.33:
                x = offset_by_upto(x, width / 16, min=5, max=width - 6)
            elif choice < 0.66:
                y = offset_by_upto(y, 3, min=6, max=13)
            else:
                breakable = not breakable
            return (x, de_type, y, breakable)

        if de_type == "5_qblock":
            y, has_powerup = de[2], de[3]
            if choice < 0.33:
                x = offset_by_upto(x, width / 16, min=5, max=width - 6)
            elif choice < 0.66:
                y = offset_by_upto(y, 3, min=6, max=13)
            else:
                has_powerup = not has_powerup
            return (x, de_type, y, has_powerup)

        if de_type == "6_stairs":
            stair_height, direction = de[2], de[3]
            if choice < 0.33:
                x = offset_by_upto(x, width / 16, min=5, max=width - 6)
            elif choice < 0.66:
                stair_height = offset_by_upto(stair_height, 2, min=1, max=5)
            else:
                direction = -direction
            return (x, de_type, stair_height, direction)

        pipe_height = de[2]
        if choice < 0.5:
            x = offset_by_upto(x, width / 16, min=5, max=width - 6)
        else:
            pipe_height = offset_by_upto(pipe_height, 2, min=2, max=4)
        return (x, de_type, pipe_height)

    def mutate(self, new_genome):
        """Use change, add, and delete mutations with a 20 percent rate."""
        new_genome = list(new_genome)

        if random.random() >= 0.20:
            heapq.heapify(new_genome)
            return new_genome

        if len(new_genome) == 0:
            new_genome.append(random_design_element())
            heapq.heapify(new_genome)
            return new_genome

        # Change operations modify existing elements; insertion and deletion
        # support variable-length genome evolution.
        operation = random.choices(
            ["change", "add", "delete"],
            weights=[60, 25, 15],
            k=1,
        )[0]

        if operation == "change":
            index = random.randrange(len(new_genome))
            new_genome[index] = self._change_element(new_genome[index])
        elif operation == "add" and len(new_genome) < 60:
            new_genome.append(random_design_element())
        elif operation == "delete" and len(new_genome) > 1:
            new_genome.pop(random.randrange(len(new_genome)))

        heapq.heapify(new_genome)
        return new_genome

    def generate_children(self, other):
        """Create two children using independent crossover points."""

        # Independent crossover points support parents with different genome
        # lengths, including empty genomes.
        point_a = random.randint(0, len(self.genome))
        point_b = random.randint(0, len(other.genome))

        child_a = self.genome[:point_a] + other.genome[point_b:]
        child_b = other.genome[:point_b] + self.genome[point_a:]

        return (
            Individual_DE(self.mutate(child_a)),
            Individual_DE(self.mutate(child_b)),
        )

    def to_level(self):
        # Cache the rendered level because the genome remains immutable after
        # individual construction.
        if self._level is None:
            base = Individual_Grid.empty_individual().to_level()

            # Render elements in a deterministic order. Later elements may
            # overwrite tiles produced by earlier overlapping elements.
            for de in sorted(self.genome, key=lambda element: (element[1], element[0], element)):
                x = de[0]
                de_type = de[1]

                if de_type == "4_block":
                    y, breakable = de[2], de[3]
                    base[y][x] = "B" if breakable else "X"

                elif de_type == "5_qblock":
                    y, has_powerup = de[2], de[3]
                    base[y][x] = "M" if has_powerup else "?"

                elif de_type == "3_coin":
                    base[de[2]][x] = "o"

                elif de_type == "7_pipe":
                    pipe_height = de[2]
                    top_y = height - pipe_height - 1
                    base[top_y][x] = "T"
                    for y in range(top_y + 1, height):
                        base[y][x] = "|"

                elif de_type == "0_hole":
                    hole_width = de[2]
                    for dx in range(hole_width):
                        base[height - 1][clip(1, x + dx, width - 2)] = "-"

                elif de_type == "6_stairs":
                    stair_height, direction = de[2], de[3]
                    for dx in range(1, stair_height + 1):
                        block_count = dx if direction == 1 else stair_height - dx
                        for dy in range(block_count):
                            row = clip(0, height - dy - 1, height - 1)
                            column = clip(1, x + dx, width - 2)
                            base[row][column] = "X"

                elif de_type == "1_platform":
                    platform_width, platform_height, material = de[2], de[3], de[4]
                    row = clip(0, height - platform_height - 1, height - 1)
                    for dx in range(platform_width):
                        column = clip(1, x + dx, width - 2)
                        base[row][column] = material

                elif de_type == "2_enemy":
                    base[height - 2][x] = "E"

            self._level = repair_grid(base)

        return self._level

    @classmethod
    def empty_individual(cls):
        return cls([])

    @classmethod
    def random_individual(cls):
        element_count = random.randint(8, 30)
        return cls([random_design_element() for _ in range(element_count)])


# ---------------------------------------------------------------------------
# Active Genome Representation
# ---------------------------------------------------------------------------

# Select the representation used for the current experiment.
# Replace Individual_Grid with Individual_DE to run the DE encoding.
Individual = Individual_Grid


# ---------------------------------------------------------------------------
# Selection and Successor Generation
# ---------------------------------------------------------------------------

def tournament_select(population, tournament_size=4):
    """Select the highest-fitness individual from a random tournament."""
    competitors = random.sample(population, min(tournament_size, len(population)))
    return max(competitors, key=Individual.fitness)


def generate_successors(population):
    """Generate the next population using elitism and tournament selection."""
    population_size = len(population)
    sorted_population = sorted(population, key=Individual.fitness, reverse=True)

    # Elitism: preserve the highest-fitness 10 percent unchanged.
    elite_count = max(1, int(population_size * 0.10))
    results = sorted_population[:elite_count]

    # Tournament selection supplies parents for crossover.
    while len(results) < population_size:
        parent_a = tournament_select(population)
        parent_b = tournament_select(population)

        children = parent_a.generate_children(parent_b)
        for child in children:
            if len(results) < population_size:
                results.append(child)

    return results


# ---------------------------------------------------------------------------
# Output and Main Evolution Loop
# ---------------------------------------------------------------------------

def write_level(filename, individual):
    with open(filename, "w") as level_file:
        for row in individual.to_level():
            level_file.write("".join(row) + "\n")


def ga():
    # Use 480 individuals for the full experiment. Smaller populations may
    # be used for local validation.
    pop_limit = 480
    os.makedirs("levels", exist_ok=True)

    # Limit the worker pool to prevent excessive memory consumption caused
    # by nested process and numerical-library parallelism.
    process_count = min(4, os.cpu_count() or 1, pop_limit)
    batch_size = int(math.ceil(pop_limit / process_count))

    with mpool.Pool(processes=process_count) as pool:
        init_time = time.time()

        # Initialize the population with mostly random individuals and a
        # small proportion of valid flat levels.
        population = [
            Individual.random_individual()
            if random.random() < 0.9
            else Individual.empty_individual()
            for _ in range(pop_limit)
        ]

        population = pool.map(
            Individual.calculate_fitness,
            population,
            batch_size,
        )

        print(
            "Created and calculated initial population statistics in:",
            time.time() - init_time,
            "seconds",
        )

        generation = 0
        start = time.time()
        print("Use ctrl-c to terminate this loop manually.")

        try:
            while True:
                best = max(population, key=Individual.fitness)
                now = time.time()

                print("Generation:", generation)
                print("Max fitness:", best.fitness())
                if generation > 0:
                    print("Average generation time:", (now - start) / generation)
                print("Net time:", now - start)

                # Store the highest-fitness individual from the current
                # generation for inspection.
                write_level("levels/last.txt", best)

                generation += 1
                next_population = generate_successors(population)
                next_population = pool.map(
                    Individual.calculate_fitness,
                    next_population,
                    batch_size,
                )
                population = next_population

        except KeyboardInterrupt:
            # Return the most recently completed population after manual
            # termination.
            pass

    return population


if __name__ == "__main__":
    # The entry-point guard is required for multiprocessing on Windows.
    final_generation = sorted(ga(), key=Individual.fitness, reverse=True)
    print("Best fitness:", final_generation[0].fitness())

    timestamp = time.strftime("%m_%d_%H_%M_%S")
    for index, individual in enumerate(final_generation[:10]):
        write_level("levels/" + timestamp + "_" + str(index) + ".txt", individual)