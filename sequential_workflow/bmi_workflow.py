# %% import dependencies
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from IPython.display import Image, display


# %% define state
class BMIstate(TypedDict):
    weight_kg: float
    height_m: float
    bmi: float
    category: str


# %% define bmi function
def calculate_bmi_fcn(state: BMIstate) -> BMIstate:
    weight = state["weight_kg"]
    height = state["height_m"]
    bmi = weight / height**2
    state["bmi"] = round(bmi, 2)
    return state


# %% Define label bmi
def label_bmi_fnc(state: BMIstate):
    bmi = state["bmi"]
    if bmi < 18.5:
        state["category"] = "Underweight"
    elif 18.5 <= bmi < 25:
        state["category"] = "Normal"
    elif 25 <= bmi < 30:
        state["category"] = "Overweight"
    else:
        state["category"] = "Obese"

    return state


# %% define graph
graph = StateGraph(BMIstate)

# %% Add nodes to graph
graph.add_node("calculate_bmi", calculate_bmi_fcn)
graph.add_node("label_bmi", label_bmi_fnc)

# %% Add edges to the graph
graph.add_edge(START, "calculate_bmi")
graph.add_edge("calculate_bmi", "label_bmi")
graph.add_edge("label_bmi", END)

# %% Compile the graph
workflow = graph.compile()

# %% Execute the graph
initial_state = {"weight_kg": 80, "height_m": 1.73}
final_state = workflow.invoke(initial_state)
print(final_state)
display(Image(workflow.get_graph().draw_mermaid_png()))
