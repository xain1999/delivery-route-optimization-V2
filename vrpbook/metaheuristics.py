"""Metaheuristics for the CVRP, from scratch (Chapter 12): adaptive large
neighborhood search (ALNS) and a compact hybrid genetic search (HGS).

Both are written for clarity first.  They are good enough to show how
the methods work and how they behave. PyVRP (Chapter 12) is the
production-grade implementation of the second.
"""
from __future__ import annotations

import math
import random
import time

import numpy as np

from .construction import clarke_wright
from .localsearch import local_search, neighbor_lists


def plan_length(routes, D) -> float:
    return sum(D[a][b] for r in routes for a, b in zip(r, r[1:]))


# ---------------------------------------------------------------- ALNS
# <<destroy>>
def random_removal(routes, q, D, rng, **_):
    customers = [c for r in routes for c in r[1:-1]]
    return rng.sample(customers, q)


def worst_removal(routes, q, D, rng, p: float = 3.0, **_):
    """Remove customers whose detour is largest, with some randomness."""
    saving = sorted(((D[r[k - 1]][r[k]] + D[r[k]][r[k + 1]] - D[r[k - 1]][r[k + 1]], r[k])
                     for r in routes for k in range(1, len(r) - 1)), reverse=True)
    chosen, pool = [], [c for _, c in saving]
    while len(chosen) < q:
        chosen.append(pool.pop(int(len(pool) * rng.random() ** p)))
    return chosen


def related_removal(routes, q, D, rng, near=None, **_):
    """Remove a customer and the customers closest to it (Shaw removal)."""
    customers = [c for r in routes for c in r[1:-1]]
    seed = rng.choice(customers)
    chosen = [seed] + [c for c in near[seed][: 3 * q] if c in set(customers)][: q - 1]
    while len(chosen) < q:                       # neighbor list too short
        c = rng.choice(customers)
        if c not in chosen:
            chosen.append(c)
    return chosen
# <<end_destroy>>


# <<repair>>
def repair(routes, removed, dist, demands, capacity, rule: str, rng) -> list[list[int]]:
    """Insert the removed customers back, one at a time: the cheapest
    insertion first ("greedy") or the largest regret first ("regret")."""
    q = np.asarray(demands, dtype=float)
    routes = [list(r) for r in routes]
    loads = [sum(demands[c] for c in r) for r in routes]
    todo = list(removed)
    rng.shuffle(todo)
    idx = np.array(todo)
    m = len(routes)
    cost = np.full((len(todo), m), np.inf)
    pos = np.zeros((len(todo), m), dtype=int)

    def refresh(r):
        rt = np.asarray(routes[r])
        a, b = rt[:-1], rt[1:]
        delta = dist[np.ix_(idx, a)] + dist[np.ix_(idx, b)] - dist[a, b]
        k = delta.argmin(axis=1)
        ok = loads[r] + q[idx] <= capacity
        cost[:, r] = np.where(ok, delta[np.arange(len(idx)), k], np.inf)
        pos[:, r] = k + 1

    for r in range(m):
        refresh(r)
    left = np.ones(len(todo), dtype=bool)
    while left.any():
        best = cost.min(axis=1)
        if np.isinf(best[left]).any():           # someone fits nowhere: open a route
            routes.append([0, 0])
            loads.append(0)
            cost = np.hstack([cost, np.full((len(todo), 1), np.inf)])
            pos = np.hstack([pos, np.zeros((len(todo), 1), dtype=int)])
            m += 1
            refresh(m - 1)
            continue
        if rule == "regret" and m > 1:
            two = np.partition(cost, 1, axis=1)
            with np.errstate(invalid="ignore"):       # rows already inserted: inf - inf
                regret = np.where(np.isinf(two[:, 1]), 1e18, two[:, 1] - two[:, 0])
            i = int(np.argmax(np.where(left, regret, -1)))
        else:
            i = int(np.argmin(np.where(left, best, np.inf)))
        r = int(np.argmin(cost[i]))
        c = todo[i]
        routes[r].insert(int(pos[i, r]), c)
        loads[r] += demands[c]
        left[i] = False
        refresh(r)
        cost[~left, :] = np.inf
    return [r for r in routes if len(r) > 2]
# <<end_repair>>


# <<alns>>
def alns(dist, demands, capacity, seconds: float, seed: int = 0, polish: bool = True,
         start_worse: float = 0.005, history: list | None = None) -> list[list[int]]:
    """Adaptive large neighborhood search with simulated-annealing acceptance
    (after Ropke and Pisinger, 2006), starting from savings."""
    rng = random.Random(seed)
    D = np.asarray(dist, dtype=float).tolist()
    near = neighbor_lists(dist, 20)
    n = len(demands) - 1
    destroys = [random_removal, worst_removal, related_removal]
    repairs = ["greedy", "regret"]
    w_d, w_r = [1.0] * len(destroys), [1.0] * len(repairs)
    s_d, s_r = [0.0] * len(destroys), [0.0] * len(repairs)
    u_d, u_r = [0] * len(destroys), [0] * len(repairs)
    current, _ = local_search(clarke_wright(dist, demands, capacity), dist, demands,
                              capacity, near=near, D=D)
    z_cur = plan_length(current, D)
    best, z_best = current, z_cur
    T0 = start_worse * z_cur / math.log(2)   # this much worse: accepted half the time
    T_end = T0 / 1000
    q_hi = min(n, max(6, min(60, n // 4)))          # never more than there are customers
    q_lo = min(q_hi, max(4, n // 20))
    t0 = time.perf_counter()
    it = 0
    while n >= 2 and (el := time.perf_counter() - t0) < seconds:
        it += 1
        T = T0 * (T_end / T0) ** (el / seconds)
        d = rng.choices(range(len(destroys)), weights=w_d)[0]
        r = rng.choices(range(len(repairs)), weights=w_r)[0]
        q = rng.randint(q_lo, q_hi)
        removed = destroys[d](current, q, D, rng, near=near)
        gone = set(removed)
        partial = [[c for c in rt if c not in gone] for rt in current]
        cand = repair(partial, removed, dist, demands, capacity, repairs[r], rng)
        if polish:                    # local search around the reinserted customers
            focus = set(removed).union(*(near[c][:5] for c in removed))
            cand, _ = local_search(cand, dist, demands, capacity, near=near, D=D,
                                   seed=it, focus=focus)
        z = plan_length(cand, D)
        score = 0.0
        if z < z_best - 1e-9:
            best, z_best, current, z_cur, score = cand, z, cand, z, 33.0
        elif z < z_cur - 1e-9:
            current, z_cur, score = cand, z, 9.0
        elif rng.random() < math.exp(-(z - z_cur) / T):
            current, z_cur, score = cand, z, 13.0
        s_d[d] += score
        s_r[r] += score
        u_d[d] += 1
        u_r[r] += 1
        if it % 100 == 0:                             # adapt the weights
            for w, s, u in ((w_d, s_d, u_d), (w_r, s_r, u_r)):
                for k in range(len(w)):
                    if u[k]:
                        w[k] = 0.9 * w[k] + 0.1 * s[k] / u[k]
                    w[k] = max(w[k], 0.05)
                    s[k], u[k] = 0.0, 0
        if history is not None and (not history or history[-1][1] > z_best):
            history.append((time.perf_counter() - t0, z_best))
    if history is not None:
        history.append((time.perf_counter() - t0, z_best, it, list(w_d), list(w_r)))
    return best
# <<end_alns>>


# ---------------------------------------------------------------- HGS
# <<split>>
def split(tour: list[int], D, demands, capacity) -> list[list[int]]:
    """Cut a giant tour into routes optimally (Prins, 2004): V[j] is the
    cost of serving the first j customers of the tour, and route
    tour[i:j] is an arc of a shortest-path problem."""
    n = len(tour)
    V = [0.0] + [math.inf] * n
    pred = [0] * (n + 1)
    for i in range(n):
        load, cost = 0, 0.0
        for j in range(i, n):
            c = tour[j]
            load += demands[c]
            if load > capacity:
                break
            cost = (D[0][c] if j == i else cost - D[tour[j - 1]][0] + D[tour[j - 1]][c]) \
                + D[c][0]
            if V[i] + cost < V[j + 1]:
                V[j + 1], pred[j + 1] = V[i] + cost, i
    routes, j = [], n
    while j > 0:
        i = pred[j]
        routes.append([0, *tour[i:j], 0])
        j = i
    return routes[::-1]
# <<end_split>>


def ox_crossover(p1: list[int], p2: list[int], rng) -> list[int]:
    """Order crossover: a slice of the first parent, the rest in the order
    of the second."""
    n = len(p1)
    a, b = sorted(rng.sample(range(n), 2))
    child = [None] * n
    child[a:b + 1] = p1[a:b + 1]
    taken = set(p1[a:b + 1])
    fill = [c for c in p2[b + 1:] + p2[:b + 1] if c not in taken]
    for k, c in zip(list(range(b + 1, n)) + list(range(0, a)), fill):
        child[k] = c
    return child


def giant_tour(routes, xy) -> list[int]:
    """Routes concatenated in the order of their angle around the depot."""
    def angle(r):
        cx = sum(xy[c][0] for c in r[1:-1]) / (len(r) - 2) - xy[0][0]
        cy = sum(xy[c][1] for c in r[1:-1]) / (len(r) - 2) - xy[0][1]
        return math.atan2(cy, cx)
    return [c for r in sorted(routes, key=angle) for c in r[1:-1]]


def _pairs(routes):
    return {frozenset(e) for r in routes for e in zip(r, r[1:])}


# <<hgs>>
def hgs(dist, demands, capacity, xy, seconds: float, seed: int = 0, mu: int = 25,
        lam: int = 40, n_elite: int = 4, n_close: int = 5,
        history: list | None = None) -> list[list[int]]:
    """A compact hybrid genetic search (after Vidal et al., 2012, 2022):
    giant tours, split, order crossover, local search as education, and
    survivor selection on cost and diversity.  Feasible plans only."""
    rng = random.Random(seed)
    D = np.asarray(dist, dtype=float).tolist()
    near = neighbor_lists(dist, 20)
    customers = list(range(1, len(demands)))
    t0 = time.perf_counter()

    def educate(routes, s):
        r, _ = local_search(routes, dist, demands, capacity, near=near, D=D, seed=s)
        return {"routes": r, "cost": plan_length(r, D), "tour": giant_tour(r, xy),
                "pairs": _pairs(r)}

    pop = [educate(clarke_wright(dist, demands, capacity), 0)]
    if len(customers) < 2:                        # nothing to cross over
        return pop[0]["routes"]
    while len(pop) < 2 or (len(pop) < mu and time.perf_counter() - t0 < 0.2 * seconds):
        tour = customers[:]
        rng.shuffle(tour)
        pop.append(educate(split(tour, D, demands, capacity), len(pop)))
    best = min(pop, key=lambda s: s["cost"])

    def distance(a, b):
        return 1 - len(a["pairs"] & b["pairs"]) / len(a["pairs"])

    def biased_fitness():
        n = len(pop)
        cost_rank = {id(s): k for k, s in enumerate(sorted(pop, key=lambda s: s["cost"]))}
        div = {}
        for s in pop:
            dd = sorted(distance(s, o) for o in pop if o is not s)
            div[id(s)] = -sum(dd[:n_close]) / max(1, min(n_close, len(dd)))
        div_rank = {id(s): k for k, s in enumerate(sorted(pop, key=lambda s: div[id(s)]))}
        return {id(s): cost_rank[id(s)] + (1 - n_elite / n) * div_rank[id(s)] for s in pop}

    fit = biased_fitness()
    it = 0
    while time.perf_counter() - t0 < seconds:
        it += 1
        a, b = (min(rng.sample(pop, 2), key=lambda s: fit[id(s)]) for _ in range(2))
        child = educate(split(ox_crossover(a["tour"], b["tour"], rng), D, demands, capacity),
                        it)
        pop.append(child)
        if child["cost"] < best["cost"] - 1e-9:
            best = child
            if history is not None:
                history.append((time.perf_counter() - t0, best["cost"]))
        if len(pop) >= mu + lam:                  # survivor selection
            while len(pop) > mu:
                fit = biased_fitness()
                clones = [s for s in pop if any(o is not s and o["cost"] == s["cost"]
                                                and o["pairs"] == s["pairs"] for o in pop)]
                worst = max(clones or pop, key=lambda s: fit[id(s)])
                pop.remove(worst)
        fit = biased_fitness() if it % 10 == 0 else {**fit, id(child): len(pop)}
    if history is not None:
        history.append((time.perf_counter() - t0, best["cost"], it))
    return best["routes"]
# <<end_hgs>>
