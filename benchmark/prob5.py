"""
benchmark with commercial nonlinear optimization solver OCTERACT
"""
import time
from pyomo.environ import ConcreteModel, Var, Reals, Objective, Constraint, minimize, SolverFactory, Binary, log

m = ConcreteModel()
m.x1 = Var(within=Reals, bounds=(0, 1.2), initialize=1)
m.x2 = Var(within=Reals, bounds=(0, 1.8), initialize=1)
m.x3 = Var(within=Reals, bounds=(0, 2.5), initialize=None)
m.y1 = Var(within=Binary, bounds=(0, 1), initialize=None)
m.y2 = Var(within=Binary, bounds=(0, 1), initialize=None)
m.y3 = Var(within=Binary, bounds=(0, 1), initialize=None)
m.y4 = Var(within=Binary, bounds=(0, 1), initialize=None)
x1 = m.x1
x2 = m.x2
x3 = m.x3
y1 = m.y1
y2 = m.y2
y3 = m.y3
y4 = m.y4

m.obj = Objective(
    expr=((x1 - 1) ** 2 + (x2 - 2) ** 2 + (x3 - 3) ** 2 + (y1 - 1) ** 2 + (y2 - 1) ** 2 + (y3 - 1) ** 2 - log(y4 + 1)),
    sense=minimize)
m.e1 = Constraint(expr=x1 + x2 + x3 + y1 + y2 + y3 <= 5.0)
m.e2 = Constraint(expr=x1 ** 2 + x2 ** 2 + x3 ** 2 + y3 ** 2 <= 5.5)
m.e3 = Constraint(expr=x1 + y1 <= 1.2)
m.e4 = Constraint(expr=x2 + y2 <= 1.8)
m.e5 = Constraint(expr=x3 + y3 <= 2.5)
m.e6 = Constraint(expr=x1 + y4 <= 1.2)
m.e7 = Constraint(expr=x2 ** 2 + y2 ** 2 <= 1.64)
m.e8 = Constraint(expr=x3 ** 2 + y3 ** 2 <= 4.25)
m.e9 = Constraint(expr=x3 ** 2 + y2 ** 2 <= 4.64)
m.write("./data/prob8.nl")
start = time.perf_counter()
SolverFactory("scip", executable="C:/Program Files/SCIPOptSuite 8.0.4/bin/scip.exe").solve(m, tee=True, keepfiles=False)
print(f"Problem 5 takes {time.perf_counter() - start} seconds to solve")
