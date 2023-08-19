class Var:
    def __init__(self, name=""):
        self.name = name
        self.lb = []
        self.ub = []
        self.x = None


class Model:
    def __init__(self, name=""):
        self.name = name
        self.var_lst = []
        self.constraint_lst = []

    def add_var(self):
        pass

    def add_constraint(self):
        """
        linear constraint is assumed
        :return:
        """
        pass
