import numpy as np


def read_xml(filename: str) -> tuple[np.array, np.array, np.array, float, int]:
    if filename.split(".")[-1] != "xml":
        raise NameError(f"Input file {filename} is not xml format")
    file = open(filename, "r")
    cx = []
    cy = []
    node_id = 0
    capacity = None
    departure_node = None
    arrival_node = None
    qty = []
    while file:
        line = next(file)
        if "node id" in line:
            node_id += 1
            cx_str = next(file)
            cx.append(value_extract(cx_str, "cx"))
            cy_str = next(file)
            cy.append(value_extract(cy_str, "cy"))

        if "departure_node" in line:
            departure_node = int(value_extract(line, "departure_node"))
        if "arrival_node" in line:
            arrival_node = int(value_extract(line, "arrival_node"))
        if "capacity" in line:
            capacity = value_extract(line, "capacity")
        if "quantity" in line:
            qty.append(value_extract(line, "quantity"))
        if "</instance>" in line:
            break
    if departure_node != arrival_node:
        print(f"Departure node {departure_node} is different from arrival node {arrival_node}")
    cx = np.array(cx)
    cy = np.array(cy)
    qty = np.array(qty)
    return cx, cy, qty, capacity, departure_node


def value_extract(line: str, string: str):
    return float(line.strip().replace(f"<{string}>", "").replace(f"</{string}>", ""))
