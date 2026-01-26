# %% Import dependencies
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from math import sqrt
from IPython.display import Image, display


# %% Create state for the graph
class quadratic(TypedDict):
    a: float
    b: float
    c: float
    d: float
    root_1: float
    root_2: float


# %% Define functions for node edges
def discriminant_cal_fcn(state: quadratic):
    d = state["b"] ** 2 - (4 * state["a"] * state["c"])
    return {"d": d}


def discriminant_ger_zero_fcn(state: quadratic):
    d_sqrt = sqrt(state["d"])
    root_1 = (-state["b"] + d_sqrt) / 2 * state["a"]

    root_2 = (-state["b"] + d_sqrt) / 2 * state["a"]

    return {"root_1": root_1, "root_2": root_2}


def discriminant_eql_zero_fcn(state: quadratic):

    root_1 = -state["b"] / (2 * state["a"])

    root_2 = -state["b"] / (2 * state["a"])

    return {"root_1": root_1, "root_2": root_2}


def discriminant_less_zero_fcn(state: quadratic):
    # Complex roots
    real_part = -state["b"] / (2 * state["a"])
    imag_part = sqrt(-state["d"]) / (2 * state["a"])
    root_1 = f"{real_part} + {imag_part}i"
    root_2 = f"{real_part} - {imag_part}i"
    return {"root_1": root_1, "root_2": root_2}


def check_condition(state: quadratic):
    if state["d"] > 0:
        return "discriminant_ger_zero"
    elif state["d"] == 0:
        return "discriminant_eql_zero"
    else:
        return "discriminant_less_zero"


# %% Create graph

graph = StateGraph(quadratic)

# Create nodes for the graphs

graph.add_node("discriminant_cal", discriminant_cal_fcn)
graph.add_node("discriminant_ger_zero", discriminant_ger_zero_fcn)
graph.add_node("discriminant_eql_zero", discriminant_eql_zero_fcn)
graph.add_node("discriminant_less_zero", discriminant_less_zero_fcn)

# create the edge of the graph

graph.add_edge(START, "discriminant_cal")
graph.add_conditional_edges("discriminant_cal", check_condition)
graph.add_edge("discriminant_ger_zero", END)
graph.add_edge("discriminant_eql_zero", END)
graph.add_edge("discriminant_less_zero", END)

# Compile the graph
workflow = graph.compile()

# %% Invoke the workflow
input = {"a": 4, "b": 2, "c": 4}

result = workflow.invoke(input)
print(result)
# %% Print the graph
display(Image(workflow.get_graph().draw_mermaid_png()))
