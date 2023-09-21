from concorde.tsp import TSPSolver


def tsp_solve(cx, cy):
    solver = TSPSolver.from_data(cx, cy, norm="EUC_2D")
    tour_data = solver.solve(verbose=False, random_seed=42)
    assert tour_data.success
    return tour_data.tour, tour_data.optimal_value
