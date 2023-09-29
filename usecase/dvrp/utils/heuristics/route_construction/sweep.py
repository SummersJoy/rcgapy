import numpy as np
from numba import njit, int32

from usecase.dvrp.utils.route.angle import get_angle
from usecase.utils.tsp import tsp_solve
from usecase.dvrp.utils.route.repr import get_route_len


class Sweep:
    def __init__(self, cx, cy, q, c, max_route_len):
        self.cx = cx
        self.cy = cy
        self.q = q
        self.c = c
        self.max_route_len = max_route_len

    def __sweep_naive(self):
        pass


def sweep_constructor(cx, cy, q, c, max_route_len, w, max_dist):
    n = len(cx) - 1
    agl = (180. - get_angle(cx, cy)) * np.pi / 180.
    agl[0] = 0.
    seq = np.argsort(agl[1:]) + 1
    best_cost = np.inf
    best_trip = np.zeros((n, max_route_len), dtype=int)
    for i in range(n):
        seq = np.concatenate((seq[i:], seq[:i]))  # rotating axis
        trip = np.zeros((n, max_route_len), dtype=int)
        cap = np.zeros(n)
        route_idx = 0
        pos = 0
        unassigned = seq.tolist().copy()
        avr = get_avr(c)
        while unassigned:
            cust = unassigned[0]
            if cap[route_idx] + q[cust] <= w:
                trip[route_idx, pos] = cust
                pos += 1
                cap[route_idx] += q[cust]
                unassigned.remove(cust)
            else:
                # step 7
                print(1)
                tour, dist, count = sweep_optimizer(trip[route_idx], cx, cy, c)
                # step 8
                jjx = np.argmin(c[tour[count - 1], [unassigned]])
                k_jjx = unassigned[jjx]
                unassigned_cpy = unassigned.copy()
                unassigned_cpy.remove(k_jjx)
                if not unassigned_cpy:
                    trip[route_idx + 1, 0] = unassigned[0]
                    break
                jji = np.argmin(c[k_jjx, [unassigned_cpy]])
                k_jji = unassigned_cpy[jji]
                ii = get_delete_loc(trip, route_idx, c, agl, avr, max_route_len)
                k_ii = trip[route_idx, ii]
                route = trip[route_idx].copy()
                route[ii] = k_jjx
                print(2)
                tour1, dist1, count = sweep_optimizer(route, cx, cy, c)
                # step 9
                if dist1 < dist:
                    # step 11
                    temp_route = unassigned[:5]
                    if k_jjx in temp_route:
                        print(3)
                        tour2, dist2, count2 = sweep_optimizer(temp_route, cx, cy, c)
                        temp_route.remove(k_jjx)
                        temp_route.append(k_ii)
                        print(4)
                        tour3, dist3, count3 = sweep_optimizer(temp_route, cx, cy, c)
                        if dist + dist2 < dist1 + dist3 and k_jji in trip[route_idx] and k_jjx in trip[route_idx]:
                            # step 13
                            tmp_route = [i for i in trip[route_idx] if i != 0]
                            tmp_route.remove(k_jji)
                            tmp_route.remove(k_jjx)
                            tmp_route.append(k_ii)
                            print(5)
                            tour4, dist4, count4 = sweep_optimizer(tmp_route, cx, cy, c)
                            if dist4 < dist and sum(q[tmp_route]) <= w:
                                # step 14
                                temp_route = unassigned[:5]
                                if k_jjx in temp_route:
                                    temp_route.remove(k_jjx)
                                if k_jji in temp_route:
                                    temp_route.remove(k_jji)
                                if k_ii not in temp_route:
                                    temp_route.append(k_ii)
                                print(6)
                                tour5, dist5, count5 = sweep_optimizer(temp_route, cx, cy, c)
                                if dist + dist2 < dist4 + dist5:
                                    # step 14 -> step 10
                                    route_idx += 1
                                    pos = 0
                                    trip[route_idx, pos] = cust
                                    pos += 1
                                    cap[route_idx] += q[cust]
                                    unassigned.remove(cust)
                                else:  # step 14 -> step 15
                                    unassigned.remove(k_jjx)
                                    unassigned.remove(k_jji)
                                    if k_jjx == cust or k_jji == cust:
                                        print("error")
                                    unassigned.append(k_ii)
                                    route = trip[route_idx]
                                    route = np.delete(route, k_ii)
                                    route = np.append(route, k_jjx)
                                    route = np.append(route, k_jji)
                                    trip[route_idx] = route[:-1]
                                    pos += 1
                                    cap[route_idx] += q[k_jjx] + q[k_jji] - q[k_ii]
                            else:
                                route_idx += 1
                                pos = 0
                                trip[route_idx, pos] = cust
                                pos += 1
                                cap[route_idx] += q[cust]
                                unassigned.remove(cust)
                            pass  # go to step 13
                        else:  # step 12
                            unassigned.remove(k_jjx)
                            if k_jjx == cust:
                                print("error")
                            unassigned.append(k_ii)
                            cap[route_idx] -= q[k_ii]
                            cap[route_idx] += q[k_jjx]
                            trip[route_idx] = np.append(np.delete(trip[route_idx], ii), 0)
                            trip[route_idx, pos - 1] = k_jjx
                            print("goto step 12")
                            pass  # goto step 12
                    else:
                        route_idx += 1
                        pos = 0
                        trip[route_idx, pos] = cust
                        pos += 1
                        cap[route_idx] += q[cust]
                        unassigned.remove(cust)
                        print("goto step 10")
                else:
                    route_idx += 1
                    pos = 0
                    trip[route_idx, pos] = cust
                    pos += 1
                    cap[route_idx] += q[cust]
                    unassigned.remove(cust)

        route_dist = 0
        for idx in range(route_idx + 1):
            t, d, count = sweep_optimizer(trip[idx], cx, cy, c)
            route_len = get_route_len(c, t[1:])
            route_dist += route_len
            trip[idx, :count] = t[1:]
        if route_dist <= best_cost:
            best_cost = route_dist
            best_trip = trip.copy()[:route_idx + 1]
    return best_trip


def sweep_step4(seq: np.array, unassigned: list, j: int):
    """
    angle increment
    """
    j += 1
    cust = seq[j]
    unassigned.remove(cust)


def sweep_step5(cap, q, cust, w, route, cx, cy, c):
    """
    capacity check
    """
    if cap + q[cust] > w:
        sweep_step7(route, cx, cy, c)


def sweep_step6(cap, q, cust):
    """
    route augmentation
    """
    cap += q[cust]


def sweep_step7(route, cx, cy, c):
    count = 0
    route_lst = [0]
    for j in route:
        if j != 0:
            route_lst.append(j)
            count += 1
    route = np.array(route_lst)
    if count > 2:
        tour, dist = tsp_solve(cx[route], cy[route])
        return route[tour], dist, count
    elif count == 1:
        return route, c[0, route[0]] * 2, count
    elif count == 2:
        d1 = c[0, route[1]] + c[route[1], route[2]] + c[route[2], 0]
        d2 = c[0, route[2]] + c[route[2], route[1]] + c[route[1], 1]
        if d1 > d2:
            return route, d1, 2
        else:
            tmp = route[2]
            route[2] = route[1]
            route[1] = tmp
            return route, d2, 2
    else:
        raise ValueError("Negative or non integral route length")


sweep_optimizer = sweep_step7


# def sweep_step8(unassigned: list, c: np.ndarray):
#     """
#     modifier
#     """
#     jjx = np.argmin(c[tour[count - 1], [unassigned]])
#     k_jjx = unassigned[jjx]


@njit
def sweep_modifier(trip, k, c, agl, avr, mtl):
    avr = get_avr(c)
    k = 0
    pos = get_delete_loc(trip, k, c, agl, avr, mtl)


def sweep_forward(cx, cy, q, c, max_route_len, w, max_dist):
    pass


@njit(fastmath=True)
def get_avr(c):
    return np.mean(c[0:])


@njit(fastmath=True)
def get_delete_loc(trip, k, c, angle, avr, mtl):
    best = np.inf
    idx = 0
    for i in range(mtl):
        cust = trip[k, i]
        if cust == 0:
            break
        val = c[cust, 0] + angle[cust] * avr
        if val < best:
            best = val
            idx = i
    return idx
