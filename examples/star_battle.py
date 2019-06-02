"""Star Battle solver example."""

from collections import defaultdict
from z3 import And, If, Implies, Sum  # type: ignore

import grilops


HEIGHT, WIDTH = 10, 10
AREAS = [
    "AAAAABBCCC",
    "AAAAABBBCC",
    "AAAAABBBBB",
    "DDDAEEBBBF",
    "DGGGEEBBFF",
    "GGGGEEBFFF",
    "GGGGGHBFFF",
    "GGGGGHBFFH",
    "JJJHHHHHHH",
    "HHHHIIIIII",
]


def main():
  """Star Battle solver example."""
  sym = grilops.SymbolSet([("EMPTY", " "), ("STAR", "*")])
  sg = grilops.SymbolGrid(HEIGHT, WIDTH, sym)

  # There must be exactly two stars per column.
  for y in range(HEIGHT):
    sg.solver.add(Sum(
        *[If(sg.cell_is(y, x, sym.STAR), 1, 0) for x in range(WIDTH)]
    ) == 2)

  # There must be exactly two stars per row.
  for x in range(WIDTH):
    sg.solver.add(Sum(
        *[If(sg.cell_is(y, x, sym.STAR), 1, 0) for y in range(HEIGHT)]
    ) == 2)

  # There must be exactly two stars per area.
  area_cells = defaultdict(list)
  for y in range(HEIGHT):
    for x in range(WIDTH):
      area_cells[AREAS[y][x]].append(sg.grid[y][x])
  for cells in area_cells.values():
    sg.solver.add(Sum(*[If(c == sym.STAR, 1, 0) for c in cells]) == 2)

  # Stars may not touch each other, not even diagonally.
  for y in range(HEIGHT):
    for x in range(WIDTH):
      sg.solver.add(Implies(
          sg.cell_is(y, x, sym.STAR),
          And(*[n.symbol == sym.EMPTY for n in sg.touching_cells(y, x)])
      ))

  if sg.solve():
    sg.print()
    if sg.is_unique():
      print("Unique solution")
    else:
      print("Alternate solution")
      sg.print()
  else:
    print("No solution")


if __name__ == "__main__":
  main()