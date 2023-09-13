def m2_cost_inter(c, r1, r2, pos1, pos2, trip):
    u_prev = trip[r1, pos1 - 1] if pos1 >= 1 else 0
    u = trip[r1, pos1]
    x = trip[r1, pos1 + 1]
    x_post = trip[r1, pos1 + 2]
    route1_break = c[u_prev, u] + c[u, x] + c[x, x_post]
    route1_repair = c[u_prev, x_post]
    route1_gain = route1_break - route1_repair
    v = trip[r2, pos2]
    v_post = trip[r2, pos2 + 1]
    route2_break = c[v, v_post]
    route2_repair = c[v, u] + c[u, x] + c[x, v_post]
    route2_gain = route2_break - route2_repair
    gain = route1_gain + route2_gain
    return gain
