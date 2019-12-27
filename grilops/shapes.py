"""This module supports puzzles that place fixed shape regions into the grid."""

import sys
from collections import defaultdict
from typing import List, Tuple
from z3 import And, ArithRef, If, Implies, Int, Or, Solver, Sum, \
               PbEq  # type: ignore

def rotate_shape_clockwise(shape):
  """Returns a new shape coordinate list rotated 90 degrees clockwise.

  # Arguments:
  shape (List[Tuple[int, int]]): A list of (y, x) coordinates defining a shape.

  # Returns:
  (List[Tuple[int, int]]): A list of (y, x) coordinates defining the 90-degree
      clockwise rotation of the input shape.
  """
  return [(x, -y) for (y, x) in shape]


def reflect_shape_y(shape):
  """Returns a new shape coordinate list reflected vertically.

  # Arguments:
  shape (List[Tuple[int, int]]): A list of (y, x) coordinates defining a shape.

  # Returns:
  (List[Tuple[int, int]]): A list of (y, x) coordinates defining the vertical
      reflection of the input shape.
  """
  return [(-y, x) for (y, x) in shape]


def reflect_shape_x(shape):
  """Returns a new shape coordinate list reflected horizontally.

  # Arguments:
  shape (List[Tuple[int, int]]): A list of (y, x) coordinates defining a shape.

  # Returns:
  (List[Tuple[int, int]]): A list of (y, x) coordinates defining the horizontal
      reflection of the input shape.
  """
  return [(y, -x) for (y, x) in shape]


def canonicalize_shape(shape):
  """Returns a new shape that's canonicalized, i.e., it's in sorted order
  and its minimum x and y values are both 0.  This helps with deduplication,
  since equivalent shapes will be canonicalized identically.

  # Arguments:
  shape (List[Tuple[int, int]]): A list of (y, x) coordinates defining a shape.

  # Returns.
  (List[Tuple[int, int]]): A list of (y, x) coordinates defining the
  canonicalized version of the shape.
  """
  min_y = min(p[0] for p in shape)
  min_x = min(p[1] for p in shape)
  canonicalized_shape = [(y - min_y, x - min_x) for (y, x) in shape]
  return sorted(canonicalized_shape)


class ShapeConstrainer:
  """Creates constraints for placing fixed shape regions into the grid.

  # Arguments
  height (int): The height of the grid.
  width (int): The width of the grid.
  shapes (List[List[Tuple[int, int]]]): A list of region shape definitions.
      Each region shape definition should be a list of (y, x) tuples.
      The same region shape definition may be included multiple times to
      indicate the number of times that shape may appear (if allow_copies
      is false).
  solver (z3.Solver, None): A #Solver object. If None, a #Solver will be
      constructed.
  complete (bool): If true, every cell must be part of a shape region. Defaults
      to false.
  allow_rotations (bool): If true, allow rotations of the shapes to be placed
      in the grid. Defaults to false.
  allow_reflections (bool): If true, allow reflections of the shapes to be
      placed in the grid. Defaults to false.
  allow_copies (bool): If true, allow any number of copies of the shapes to be
      placed in the grid. Defaults to false.
  max_num_instances (int, None): An upper bound on the total number of shape
      occurences in the grid. Ignored if allow_copies is false, since in that
      case the exact number of instances is known. If None, no upper bound is
      assumed. Defaults to None.
  """
  _instance_index = 0

  def __init__(  # pylint: disable=R0913
      self,
      height: int,
      width: int,
      shapes: List[List[Tuple[int, int]]],
      solver: Solver = None,
      complete: bool = False,
      allow_rotations: bool = False,
      allow_reflections: bool = False,
      allow_copies: bool = False,
      max_num_instances: int = None
  ):
    ShapeConstrainer._instance_index += 1
    if solver:
      self.__solver = solver
    else:
      self.__solver = Solver()

    self.__complete = complete
    self.__allow_copies = allow_copies

    if self.__allow_copies:
      if max_num_instances is None:
        max_shape_size = min(len(shape) for shape in shapes)
        self.__max_num_instances = (height * width) // max_shape_size
      else:
        self.__max_num_instances = max_num_instances
    else:
      self.__max_num_instances = len(shapes)

    self.__shapes = shapes
    self.__variants = self.__make_variants(allow_rotations, allow_reflections)
    self.__make_placements(height, width)

    self.__create_grids(height, width)
    self.__create_instances()
    self.__add_constraints()

  def __make_variants(self, allow_rotations, allow_reflections):
    all_variants = []
    for shape in self.__shapes:
      variants = [shape]
      if allow_rotations:
        for _ in range(3):
          variants.append(rotate_shape_clockwise(variants[-1]))
      if allow_reflections:
        more_variants = []
        for variant in variants:
          more_variants.append(variant)
          more_variants.append(reflect_shape_y(variant))
          more_variants.append(reflect_shape_x(variant))
        variants = more_variants
      variants = [
        list(s)
        for s in {tuple(canonicalize_shape(s)) for s in variants}
      ]
      all_variants.append(variants)
    return all_variants

  def __make_placements(self, height, width):
    """Create the list of list of placements, one list of placements
    for each shape.
    """
    self.__placements = []
    for shape_type in range(len(self.__shapes)):
      placements = []
      for variant_index, variant in enumerate(self.__variants[shape_type]):
        max_dy = max(p[0] for p in variant)
        max_dx = max(p[1] for p in variant)
        unfitting_y = height - max_dy
        unfitting_x = width - max_dx
        for y in range(unfitting_y):
          for x in range(unfitting_x):
            placements.append((y, x, variant_index))
      self.__placements.append(placements)

  def __create_grids(self, height: int, width: int):
    """Create the grids used to model shape region constraints."""
    self.__shape_instance_grid: List[List[ArithRef]] = []
    for y in range(height):
      row = []
      for x in range(width):
        v = Int(f"scsi-{ShapeConstrainer._instance_index}-{y}-{x}")
        if self.__complete:
          self.__solver.add(v >= 0)
        else:
          self.__solver.add(v >= -1)
        self.__solver.add(v < height * width)
        row.append(v)
      self.__shape_instance_grid.append(row)

    if self.__allow_copies:
      self.__shape_type_grid: List[List[ArithRef]] = []
      for y in range(height):
        row = []
        for x in range(width):
          v = Int(f"scst-{ShapeConstrainer._instance_index}-{y}-{x}")
          if self.__complete:
            self.__solver.add(v >= 0)
          else:
            self.__solver.add(v >= -1)
          self.__solver.add(v < len(self.__shapes))
          row.append(v)
        self.__shape_type_grid.append(row)

  def __create_instances(self):
    """Create arrays of variables representing the shape instances."""

    # The instance-placement array is an array of variables,
    # one for each shape instance.  It indicates which placement
    # of that instance is used in the grid.  A "placement"
    # is a per-shape ID representing a specific variant of that
    # shape and a specific position in the grid where that
    # variant is placed.

    self.__instance_placements: List[ArithRef] = [
        Int(f"scip-{ShapeConstrainer._instance_index}-{instance_index}")
        for instance_index in range(self.__max_num_instances)
    ]

    # The instance-shapes array is an array of variables, one for each shape
    # instance.  It indicates which shape is used for that instance (or -1 if
    # the instance doesn't correspond to a shape). There's no need for an
    # instance-shapes array when allow_copies is false, since in that case it's
    # clear which shape is used for each instance: shape i is used for instance
    # i.

    if self.__allow_copies:
      self.__instance_shapes: List[ArithRef] = [
        Int(f"scis-{ShapeConstrainer._instance_index}-{instance_index}")
        for instance_index in range(self.__max_num_instances)
      ]

  def __add_constraints(self):
    self.__add_instance_array_constraints()
    self.__add_grid_constraints()

    for shape_type in range(len(self.__shapes)):
      self.__add_shape_constraints(shape_type)

  def __add_instance_array_constraints(self):
    for instance_index in range(self.__max_num_instances):
      placement = self.__instance_placements[instance_index]
      if not self.__allow_copies:
        self.__solver.add(placement >= 0)
        self.__solver.add(placement < len(self.__placements[instance_index]))
        continue
      
      shape = self.__instance_shapes[instance_index]
      self.__solver.add(shape >= -1)
      self.__solver.add(shape < len(self.__shapes))
      self.__solver.add(placement >= -1)
      self.__solver.add((shape == -1) == (placement == -1))
      for shape_type in range(len(self.__shapes)):
        self.__solver.add(
            Implies(
                shape == shape_type,
                placement < len(self.__placements[shape_type])
            )
        )

      # To reduce non-determinism, we force instances to be ordered by shape
      # index, followed by placement.

      if instance_index > 0:
        prev_placement = self.__instance_placements[instance_index - 1]
        prev_shape = self.__instance_shapes[instance_index - 1]
        self.__solver.add(
            Or(shape == -1, And(shape >= prev_shape, prev_shape != -1))
        )
        self.__solver.add(
            Implies(shape == prev_shape, placement >= prev_placement)
        )

  def __add_grid_constraints(self):
    for y in range(len(self.__shape_instance_grid)):
      for x in range(len(self.__shape_instance_grid[0])):
        grid_instance = self.__shape_instance_grid[y][x]
        if self.__complete:
          self.__solver.add(grid_instance >= 0)
        else:
          self.__solver.add(grid_instance >= -1)
        self.__solver.add(grid_instance < self.__max_num_instances)

        if self.__allow_copies:
          grid_shape = self.__shape_type_grid[y][x]
          self.__solver.add((grid_instance == -1) == (grid_shape == -1))
          for instance_index in range(self.__max_num_instances):
            self.__solver.add(
                Implies(
                    grid_instance == instance_index,
                    grid_shape == self.__instance_shapes[instance_index]
                )
            )

  def __add_shape_constraints(self, shape_type):
    valid_placements_by_coord = defaultdict(list)

    for placement_index, placement in enumerate(self.__placements[shape_type]):
      (ly, lx, variant_index) = placement
      grid_instances = []
      variant = self.__variants[shape_type][variant_index]
      for (y, x) in ((ly + dy, lx + dx) for (dy, dx) in variant):
        valid_placements_by_coord[(y, x)].append(placement_index)
        grid_instances.append(self.__shape_instance_grid[y][x])

      if self.__allow_copies:
        for instance_index in range(self.__max_num_instances):
          grid_constraints = [instance == instance_index for instance in grid_instances]
          self.__solver.add(
              Implies(
                  And(
                      self.__instance_shapes[instance_index] == shape_type,
                      self.__instance_placements[instance_index] == placement_index
                  ),
                  And(*grid_constraints)
              )
          )
      else:
        grid_constraints = [instance == shape_type for instance in grid_instances]
        self.__solver.add(
            Implies(
                self.__instance_placements[shape_type] == placement_index,
                And(*grid_constraints)
            )
        )

    for y in range(len(self.__shape_instance_grid)):
      for x in range(len(self.__shape_instance_grid[0])):
        if self.__allow_copies:
          for instance_index in range(self.__max_num_instances):
            placement = self.__instance_placements[instance_index]
            self.__solver.add(
                Implies(
                    And(
                        self.__shape_type_grid[y][x] == shape_type,
                        self.__shape_instance_grid[y][x] == instance_index
                    ),
                    Or(*[
                       (placement == p) for p in valid_placements_by_coord[(y, x)]
                    ])
                )
            )
        else:
          placement = self.__instance_placements[shape_type]
          self.__solver.add(
              Implies(
                  self.__shape_instance_grid[y][x] == shape_type,
                  Or(*[
                     placement == p for p in valid_placements_by_coord[(y, x)]
                  ])
              )
          )

  @property
  def solver(self) -> Solver:
    """(z3.Solver): The #Solver associated with this #ShapeConstrainer."""
    return self.__solver

  @property
  def shape_type_grid(self) -> List[List[ArithRef]]:
    """(List[List[ArithRef]]): A grid of z3 constants of shape types.

    Each cell contains the index of the shape type placed in that cell (as
    indexed by the shapes list passed in to the #ShapeConstrainer constructor),
    or -1 if no shape is placed within that cell.
    """
    if self.__allow_copies:
      return self.__shape_type_grid
    else:
      return self.__shape_instance_grid

  @property
  def shape_instance_grid(self) -> List[List[ArithRef]]:
    """(List[List[ArithRef]]): A grid of z3 constants of shape instance IDs.

    Each cell contains a number shared among all cells containing the same
    instance of the shape, or -1 if no shape is placed within that cell.
    """
    return self.__shape_instance_grid

  def print_shape_types(self):
    """Prints the shape type assigned to each cell.

    Should be called only after the solver has been checked.
    """
    model = self.__solver.model()
    for row in self.shape_type_grid:
      for v in row:
        shape_index = model.eval(v).as_long()
        if shape_index >= 0:
          sys.stdout.write(f"{shape_index:3}")
        else:
          sys.stdout.write("   ")
      print()

  def print_shape_instances(self):
    """Prints the shape instance ID assigned to each cell.

    Should be called only after the solver has been checked.
    """
    model = self.__solver.model()
    for row in self.shape_instance_grid:
      for v in row:
        shape_index = model.eval(v).as_long()
        if shape_index >= 0:
          sys.stdout.write(f"{shape_index:3}")
        else:
          sys.stdout.write("   ")
      print()

    print("FOR DEBUGGING:")
    for instance_index in range(self.__max_num_instances):
      if self.__allow_copies:
        shape = model.eval(self.__instance_shapes[instance_index]).as_long()
      else:
        shape = instance_index
      placement = model.eval(self.__instance_placements[instance_index]).as_long()
      print(f"Instance {instance_index} has shape {shape} and placement {placement}")
      if shape != -1 and placement != -1:
        (ly, lx, variant_index) = self.__placements[shape][placement]
        print(f"... with ly={ly}, lx={lx}, variant={variant_index}")
        variant = self.__variants[shape][variant_index];
        print(f"... with {variant}")
