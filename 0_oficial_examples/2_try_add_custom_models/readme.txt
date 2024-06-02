Aunque se cree una clase que herede los métodos de una regresión lineal, por ejemplo,

class lr_custom(LinearRegression)

Y se modifique el código para tener una regresión lineal custom, el código se puede entrenar y se reconoce como una instancia de la clase que fue heradado,

PERO GUROBI DE TODAS FORMAS NO LO RECONOCE Y POR LO TANTO NO ES POSIBLE TENER UN MODELO CUSTOM, DE NINGUNA FORMA


obs: la única opción podría ser tener 2 modelos y tener un if para hacer switch, no se va a poder empaquetar en el modelo y va a tener que el codigo
del optimizador


from gurobipy import *

# Crear el modelo
m = Model()

# Crear variables
x = m.addVar(vtype=GRB.CONTINUOUS, name="x")

# Actualizar el modelo para agregar las variables
m.update()

# Agregar restricciones condicionales
if x > 0.5:
    m.addConstr(a >= 1, "restriccion1")
else:
    m.addConstr(a >= 2, "restriccion2")

# Optimizar el modelo
m.optimize()