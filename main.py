from cvrp import CVRP, Heuristicas
import numpy as np
import random as rd

if __name__ == '__main__':
    if __debug__:
        print('modo debug ATIVADO')
    else:
        print('mode debug DESATIVADO')
    np.random.seed(7)
    rd.seed(7)
    
    cvrp = CVRP('instancias/instancia1-vrp.txt')

    heuristicas = Heuristicas(cvrp, plot=False)

    print(cvrp)

    # cost, route = heuristicas.ant_colony(ite=1, ants=1000, online=True, elitist=True, evapor=0.3)
    cost, route = heuristicas.ant_colony(ite=50, ants=20, online=False, update_by='rank', k=5, worst=True, elitist=True, evapor=0.5)
    # cost, route = heuristicas.ant_colony(ite=50, ants=20, online=False, update_by='quality', k=5, worst=True, elitist=True, evapor=0.3)

    print(cvrp.route_cost(route))
    cvrp.plot(routes=route)
