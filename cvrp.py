import itertools
import numpy as np
import random as rd
import os
import tsp
import time
import networkx as nx
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist, squareform
from copy import deepcopy
from builtins import property, reversed


def timeit(f):
    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % (f.__name__, args, kw, te - ts))
        return result

    return timed

def progress(done, total, text: str):
    x = int(round(40.0 * done / total))
    print(f"\r{text}: |{'█' * x}{'-' * (40 - x)}|", end='')
    if done == total:
        print()
    pass

class CVRP:
    """
    Representa uma instância de um problema de roteamento de veículos capacitado
    """
    _graph = None

    @property
    def graph(self):
        if self._graph is None:
            self._graph = nx.DiGraph()
            self._graph.add_nodes_from(range(self.n))
        return self._graph

    _c = None

    @property
    def c(self):
        """ matriz de distâncias"""
        if self._c is None:
            self._c = np.round(np.matrix(data=squareform(pdist(self.coord))))
        return self._c

    def __str__(self):
        return self.info['NAME']

    def __init__(self, path: str):
        """
        :param path: Arquivo no formato cvrp da CVRPLIB
        """
        assert os.path.exists(path), path + ' - arquivo não existe.'
        with open(path, 'r') as f:
            self.info = {}
            for ln in f:
                if ln.strip() == 'NODE_COORD_SECTION':
                    break
                self.info[ln.split(':')[0].strip()] = ln.split(':')[1].strip()
            assert self.info['EDGE_WEIGHT_TYPE'] == 'EUC_2D', 'tipo de distância não suportado: ' + self.info[
                'EDGE_WEIGHT_TYPE']
            self.q = int(self.info['CAPACITY'])
            """Capacidade"""
            self.n = int(self.info['DIMENSION'])
            """Número de pontos"""
            self.k = int(self.info['NAME'].split('-k')[-1])
            """Número mínimo de rotas"""
            self.coord = np.zeros(shape=[self.n, 2], dtype=float)
            """Coordenadas no formato matriz nx2"""
            for i in range(self.n):
                v = f.readline().split()
                self.coord[i][0] = float(v[1])
                self.coord[i][1] = float(v[2])
            for ln in f:
                if ln.strip() == 'DEMAND_SECTION':
                    break
            self.d = np.zeros(self.n, dtype=int)
            """Demandas"""
            for i in range(self.n):
                v = f.readline().split()
                self.d[i] = int(v[1])
        pass

    def plot(self, routes=None, edges=None, clear_edges=True, stop=True, sleep_time=0.01):
        """
        Exibe a instância graficamente

        :param routes: Solução (lista de listas)
        :param edges: lista de arcos (lista de tuplas (i,j) )
        :param clear_edges: limpar o último plot ou não
        :param stop: Parar a execução ou não
        :param sleep_time: Se stop for Falso, tempo de espera em segundos antes de prosseguir
        """
        if clear_edges:
            self.graph.clear_edges()
        if routes is not None:
            for r in routes:
                if len(r) > 1:
                    for i in range(len(r) - 1):
                        self.graph.add_edge(r[i], r[i + 1])
                    self.graph.add_edge(r[-1], r[0])
        if edges is not None:
            for i, j in edges:
                self.graph.add_edge(i, j)
        plt.clf()
        color = ['#74BDCB' for i in range(self.n)]
        color[0] = '#FFA384'
        nx.draw_networkx(self.graph, self.coord, with_labels=True, node_size=120, font_size=8, node_color=color)
        if stop:
            plt.show()
        else:
            plt.draw()
            plt.pause(sleep_time)
        pass

    def route_cost(self, routes):
        """
        Calcula o custo da solução

        :param routes: Solução (lista de listas)
        :return : float custo total
        """
        cost = 0
        for r in routes:
            for i in range(1, len(r)):
                cost += self.c[r[i - 1], r[i]]
            cost += self.c[r[-1], r[0]]
        return cost

    def is_feasible(self, routes):
        """
        Verifica se as restrições do problema foram satisfeitas ou não

        :param routes: Solução (lista de listas)
        :return : bool True se for uma solução viável
        """
        if max([self.d[r].sum() for r in routes]) > self.q:
            print("capacidade violada")
            return False
        count = np.zeros(self.n, dtype=int)
        for r in routes:
            for i in r:
                count[i] += 1
        if max(count[1:]) > 1:
            print("cliente vizitado mais de uma vez")
            return False
        if min(count[1:]) < 1:
            print("cliente não vizitado")
            return False
        return True

class Heuristicas():
    """
    Classe com método heurísticos para o CVRP
    """
    _saving = None

    @property
    def saving(self):
        """
        Matriz de valores de 'savings' (c[i, 0] + c[0, j] - c[i, j])
        """
        if self._saving is None:
            c = self.cvrp.c
            n = self.cvrp.n
            s = np.zeros(shape=[n, n])
            for i in range(1, n):
                for j in range(1, i):
                    s[i, j] = c[i, 0] + c[0, j] - c[i, j]
                    s[j, i] = c[j, 0] + c[0, i] - c[j, i]
            self._saving = s
        return self._saving

    def __init__(self, cvrp: CVRP, plot=False):
        """
        :param cvrp: Instância de um CVRP
        :param plot: Se as soluções parciais devem ser exibidas ou não
        """
        self.cvrp = cvrp
        self.plot = plot
        pass

    _max_saving = None
    
    @property
    def max_saving(self):
        if self._max_saving is None:
            s = self.saving
            self._max_saving = [s[i, :].max() for i in range(len(s))]
        return self._max_saving

    @timeit
    def Clarke_n_Wright(self, routes=None):
        """
        Aplica o algoritmo de Clarke and Wright paralelo

        :param routes: Solução (lista de listas), caso seja passada uma solução,
        o algoritmo se ocupa de tentar mesclar as rotas existentes nesta solução.
        :return : tupla (custo, solução)
        """
        n = self.cvrp.n
        d = self.cvrp.d
        q = self.cvrp.q

        # cria n rotas triviais
        if routes is None:
            routes = [[0, i] for i in range(1, n)]
        else:
            for i in reversed(range(len(routes))):
                if len(routes[i]) <= 1:
                    del routes[i]

        load_r_zipped = [[d[r].sum(), r] for r in routes]
        # calcular os 'savings'
        s = self.saving

        cost = self.cvrp.route_cost(routes)
        # concatenar rotas
        max_s = self.max_saving
        while True:
            argmax = None
            max_val = 0
            load_r_zipped.sort(key=lambda a: max_s[a[1][-1]], reverse=True)
            for k, rk in enumerate(load_r_zipped):
                if max_s[rk[1][-1]] <= max_val:
                    break
                for l, rl in enumerate(load_r_zipped):
                    if (k != l) and max_val < s[rk[1][-1], rl[1][1]] and rk[0] + rl[0] <= q:
                        argmax = k, l
                        max_val = s[rk[1][-1], rl[1][1]]

            if argmax is not None:
                # concatenar
                k, l = argmax
                cost -= s[load_r_zipped[k][1][-1], load_r_zipped[l][1][1]]
                load_r_zipped[k][1].extend(load_r_zipped[l][1][1:])
                load_r_zipped[l][1].clear()
                load_r_zipped[k][0] += load_r_zipped[l][0]
                del load_r_zipped[l]
                if self.plot:
                    self.cvrp.plot(routes=routes, clear_edges=True, stop=False)
            else:
                break

        # remover rotas vazias
        for i in reversed(range(len(routes))):
            if len(routes[i]) <= 1:
                del routes[i]

        assert self.cvrp.is_feasible(routes)
        assert cost == self.cvrp.route_cost(routes)
        return cost, routes

    def intra_route(self, route, cost=0):
        chg = False
        for r in route:
            imp = True
            while imp:
                imp = tsp.two_opt(r, self.cvrp.c)
                if not imp:
                    imp = tsp.three_opt(r, self.cvrp.c)
                if imp:
                    chg = True
            if self.plot:
                self.cvrp.plot(routes=route, clear_edges=True, stop=False)
        if chg:
            cost = self.cvrp.route_cost(route)
        assert self.cvrp.is_feasible(route)
        return chg, cost

    def _arg_best_insection(self, route, v):
        c = self.cvrp.c
        n = len(route)
        min_arg = n
        min_val = c[route[-1], v] + c[v, route[0]] - c[route[-1], route[0]]
        for i in range(1, n):
            d = c[route[i - 1], v] + c[v, route[i]] - c[route[i - 1], route[i]]
            if d < min_val:
                min_val = d
                min_arg = i
        return min_arg, min_val

    def replace(self, route, cost=0):
        q = self.cvrp.q
        c = self.cvrp.c
        d = self.cvrp.d
        chg = False
        imp = True
        load = [d[r].sum() for r in route]
        while imp:
            imp = False
            for a, ra in enumerate(route):
                for i, vi in enumerate(ra):
                    if i == 0:
                        continue

                    rem_cost = c[ra[i - 1], ra[(i + 1) % len(ra)]] - c[ra[i - 1], ra[i]] - c[
                        ra[i], ra[(i + 1) % len(ra)]]
                    if rem_cost > -1e-3:
                        continue
                    min_val = np.inf
                    min_arg = None
                    for b, rb in enumerate(route):
                        if load[b] + d[vi] <= q and a != b:
                            insert_pos, add_cost = self._arg_best_insection(rb, vi)
                            if add_cost < min_val and add_cost + rem_cost < -1e-3:
                                min_val = add_cost
                                min_arg = b, insert_pos
                                if min_val < 1e-3:
                                    break
                    if min_arg is not None and min_val + rem_cost < -1e-3:
                        del ra[i]
                        load[a] -= d[vi]
                        route[min_arg[0]].insert(min_arg[1], vi)
                        load[min_arg[0]] += d[vi]
                        chg = imp = True
                        cost += min_val + rem_cost
                        if self.plot:
                            self.cvrp.plot(routes=route + [ra], clear_edges=True, stop=False)
                        break
        assert self.cvrp.is_feasible(route)
        return chg, cost

    def swap(self, route, cost=0):
        q = self.cvrp.q
        c = self.cvrp.c
        d = self.cvrp.d
        imp = True
        chg = False
        load = [d[r].sum() for r in route]
        while imp:
            imp = False
            for a in range(1, len(route)):
                ra = route[a]
                for i in range(1, len(ra)):
                    vi = ra[i]
                    for b in range(a):
                        rb = route[b]
                        for j in range(1, len(rb)):
                            vj = rb[j]
                            if load[a] + d[vj] - d[vi] <= q and load[b] + d[vi] - d[vj] <= q:
                                delta = c[ra[i - 1], vj] + c[vj, ra[(i + 1) % len(ra)]] - c[ra[i - 1], vi] - \
                                        c[vi, ra[(i + 1) % len(ra)]] + c[rb[j - 1], vi] + c[vi, rb[(j + 1) % len(rb)]] - \
                                        c[rb[j - 1], vj] - c[vj, rb[(j + 1) % len(rb)]]
                                if delta < -1e-3:
                                    ra[i] = vj
                                    rb[j] = vi

                                    load[a] += d[vj] - d[vi]
                                    load[b] += d[vi] - d[vj]
                                    chg = imp = True
                                    vi, vj = vj, vi
                                    cost += delta
                                    if self.plot:
                                        self.cvrp.plot(routes=route + [ra], clear_edges=True, stop=False)
        assert self.cvrp.is_feasible(route)
        return chg, cost

    def two_opt_star(self, route, cost=0):
        q = self.cvrp.q
        c = self.cvrp.c
        d = self.cvrp.d
        imp = True
        chg = False
        while imp:
            imp = False
            for a in range(1, len(route)):
                ra = route[a]
                if len(ra) < 3:
                    continue
                for i in range(1, len(ra)):
                    vi = ra[i]
                    vni = ra[(i + 1) % len(ra)]
                    for b in range(a):
                        rb = route[b]
                        if len(rb) < 3:
                            continue
                        for j in range(1, len(rb)):
                            vj = rb[j]
                            vnj = rb[(j + 1) % len(rb)]
                            delta = c[vj, vni] + c[vi, vnj] - c[vi, vni] - c[vj, vnj]
                            if delta < -1e-3:
                                if sum(d[ra[0:i + 1]]) + sum(d[rb[j + 1:]]) <= q and sum(d[rb[0:j + 1]]) + sum(
                                        d[ra[i + 1:]]) <= q:
                            
                                    na = ra[0:i + 1] + rb[j + 1:]
                                    nb = rb[0:j + 1] + ra[i + 1:]
                                    ra.clear()
                                    ra.extend(na)
                                    rb.clear()
                                    rb.extend(nb)
                                    chg = imp = True
                                    cost += delta
                                    if self.plot:
                                        self.cvrp.plot(routes=route + [ra], clear_edges=True, stop=False)
                                    break
                            delta = c[vnj, vni] + c[vi, vj] - c[vi, vni] - c[vj, vnj]
                            if delta < -1e-3:
                                if sum(d[ra[:i + 1]]) + sum(d[rb[:j + 1]]) <= q and sum(d[rb[j + 1:]]) + sum(
                                        d[ra[i + 1:]]) <= q:
                                    
                                    na = ra[:i + 1] + rb[j:0:-1]
                                    nb = [0] + ra[:i:-1] + rb[j + 1:]
                                    ra.clear()
                                    ra.extend(na)
                                    rb.clear()
                                    rb.extend(nb)
                                    chg = imp = True
                                    cost += delta
                                    if self.plot:
                                        self.cvrp.plot(routes=route + [ra], clear_edges=True, stop=False)
                                    break

                        if imp:
                            break
                    if imp:
                        break
                if imp:
                    break

        assert self.cvrp.is_feasible(route)
        return chg, cost

    def VND(self, sol, cost=None):
        """
        Variable Neighborhood Descent
        :param sol: Solução (lista de listas)
        :param cost: Custo atual da solução
        :return: tupla (custo, solução)
        """
        if cost is None:
            cost = self.cvrp.route_cost(sol)
        imp = True
        while imp:
            np.random.shuffle(sol)
            imp = False
            if not imp:
                imp, cost = self.swap(sol, cost)
            if not imp:
                imp, cost = self.replace(sol, cost)
            if not imp:
                imp, cost = self.two_opt_star(sol, cost)
            if not imp:
                imp, cost = self.intra_route(sol, cost)

        # eliminar rotas vazias
        for i in reversed(range(len(sol))):
            if len(sol[i]) <= 1:
                del sol[i]

        assert self.cvrp.is_feasible(sol)
        assert cost == self.cvrp.route_cost(sol)
        return cost, sol

    def _ant_run(self, trail):

        n = self.cvrp.n
        d = self.cvrp.d
        q = self.cvrp.q
        c = self.cvrp.c
        sol = []

        maxc = c.max()

        visited = np.zeros([n], dtype=bool)

        cont = 1
        while cont < n:
            path = [0]
            v = 0
            load = float(0)
            while True:
                can = [i for i in range(n) if not visited[i] and load + d[i] <= q and v != i]
                if len(can) == 0:
                    break
                weight = np.array([max(trail[v, i], self._min_trail) for i in can])

                # heuristica
                heu = np.array([(maxc - c[v, i]) / maxc for i in can])
                if v != 0:
                    if load < q * 0.5:
                        heu *= np.array([2 if c[0, i] > c[0, v] else 1 for i in can])
                    else:
                        heu *= np.array([2 if c[0, i] < c[0, v] else 1 for i in can])

                heu /= heu.max()
                weight /= weight.max()  # normalizar

                weight = weight * heu

                v = rd.choices(can, weights=weight)[0]
                if v == 0:
                    break
                else:
                    path.append(v)
                    load += d[v]
                    visited[v] = True
                    cont += 1
            sol.append(path)

        return sol

    _min_trail = 0.001

    def _reinforcement(self, sol, valor, trail):
        c = self.cvrp.c
        for r in sol:
            if c[r[0], r[1]] < c[r[-1], r[0]]:
                for i in range(1, len(r)):
                    trail[r[i - 1], r[i]] += valor
                trail[r[-1], r[0]] += valor
            else:
                for i in range(1, len(r)):
                    trail[r[i], r[i - 1]] += valor
                trail[r[0], r[-1]] += valor

    def _plot_trail(self, trail: np.matrix):
        G = self.cvrp.graph
        G.clear_edges()
        maxw = trail.max() / 2
        for i, j in itertools.permutations(range(len(trail)), 2):
            if trail[i, j] > 0:
                G.add_edge(i, j, weight=trail[i, j] / maxw)
        weights = list(nx.get_edge_attributes(G, 'weight').values())
        plt.clf()
        nx.draw(G, self.cvrp.coord, with_labels=True, node_size=120, font_size=8, width=weights)
        plt.draw()
        plt.pause(.01)
    
    @timeit
    def ant_colony(self, ite: int, ants: int, evapor=0.1, online=True, update_by='quality', k=1, worst=False,
                   elitist=False):
        """
        Ant Colony Optimization

        :param ite: número de iterações
        :param ants: número de formigas
        :param evapor: taxa de evaporação
        :param online:
            True - a trilha é atualizada quando cada formiga termina seu percurso (Online delayed pheromone update);
            False - a trilha é atualizada apenas após todas as formigas terminarem seu percurso (offline)
        :param update_by:
            Usado quando online == False
            'quality' - as formigas que geraram as k melhores soluções depositam um valor constante às respectivas trilhas.
            'rank' - as formigas que geraram as k melhores soluções depositam um valor relativo as seu rank às respectivas trilhas.
        :param worst: True -  a formiga que gerou a pior solução decrementa o feromônio  da sua trilha
        :param elitist: True- a melhor solução até então gerada adiciona feromônio à sua trilha
        :return:tupla (custo, solução)
        """
        n = self.cvrp.n
        trail = np.zeros(shape=[n, n], dtype=float)
        best_route = None
        best_cost = np.inf

        if online:
            # online delayed update
            best_cost, best_route = self.Clarke_n_Wright()
            print(f'\n{0} AC  {best_cost} ')
            UB = best_cost * 2
            ite_ants = ite * ants
            for i in range(ite_ants):
                sol = self._ant_run(trail)
                cost = self.cvrp.route_cost(sol)
                progress(i + 1, ite_ants, f'Ant: {i + 1} \tCost: {cost} \t Best: {best_cost}')
                cost, sol = self.VND(sol, cost)
                if cost < best_cost:
                    best_cost = cost
                    best_route = deepcopy(sol)
                    print(f'\n{i + 1} AC  {best_cost} ')
                # evaporação
                trail = (1 - evapor) * trail
                # reforço
                delta = (UB - cost) / UB
                self._reinforcement(sol, delta, trail)
                if elitist:
                    self._reinforcement(best_route, delta, trail)

        else:
            # offline update
            for i in range(ite):
                lista = []
                for f in range(ants):
                    sol = self._ant_run(trail)
                    cost = self.cvrp.route_cost(sol)
                    progress(f + 1, ants, f'Turno: {i + 1} \tLast Ant: {cost} \t Best: {best_cost}')
                    cost, sol = self.VND(sol, cost)
                    lista.append((cost, sol))
                    if cost < best_cost:
                        best_cost = cost
                        best_route = deepcopy(sol)

                # evaporação
                trail = (1 - evapor) * trail

                # reforço
                if worst:
                    cost, sol = max(lista)
                    self._reinforcement(sol, -1, trail)

                if elitist:
                    self._reinforcement(best_route, 1, trail)

                if update_by == 'quality':
                    lista.sort()
                    for cost, sol in lista[:k]:
                        self._reinforcement(sol, 1, trail)
                elif update_by == 'rank':
                    lista.sort()
                    delta = k
                    for cost, sol in lista[:k]:
                        self._reinforcement(sol, delta, trail)
                        delta -= 1
                self._plot_trail(trail)

        return best_cost, best_route
