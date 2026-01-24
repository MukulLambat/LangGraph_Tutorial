# %% Import dependencies
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_openai import ChatOpenAI
from IPython.display import Image, display

# %% Define LLM instance
llm = ChatOpenAI(
    base_url="http://127.0.0.1:8080",
    api_key="not required",
    model="local_model",
)
# response = llm.invoke("hello").content
# print(response)


# %% State of the workflow
class cricket_stats(TypedDict):
    runs: int
    balls: int
    fours: int
    sixes: int

    strike_rate: float
    runs_in_boundary_per: float
    balls_per_boundary: int
    summary: str


# %% Define functions for the node execution
def strike_rate_fcn(state: cricket_stats):
    runs = state["runs"]
    balls = state["balls"]
    strike_rate = (runs / balls) * 100
    # state["strike_rate"] = strike_rate
    return {"strike_rate": strike_rate}


def runs_in_boundary_fcn(state: cricket_stats):
    fours = state["fours"]
    runs = state["runs"]
    runs_in_fours = ((fours * 4) / runs) * 100
    # state["runs_in_boundary_per"] = runs_in_fours
    return {"runs_in_fours": runs_in_fours}


def balls_per_boundary_fcn(state: cricket_stats):
    balls = state["balls"]
    fours = state["fours"]
    sixes = state["sixes"]
    balls_per_boundary = balls / (fours + sixes)
    state["balls_per_boundary"] = balls_per_boundary
    return {"balls_per_boundary": balls_per_boundary}


def summary_fcn(state: cricket_stats):
    prompt = f"summarize the result in easily understandable way. \n\n {state}"
    summary = llm.invoke(prompt)
    state["summary"] = summary
    return state


# %% Define Graph
graph = StateGraph(cricket_stats)

# Define nodes of the graphs
graph.add_node("strike_rate", strike_rate_fcn)
graph.add_node("runs_in_boundary", runs_in_boundary_fcn)
graph.add_node("balls_per_boundary", balls_per_boundary_fcn)
graph.add_node("summary", summary_fcn)

# Create edges of the graph
graph.add_edge(START, "strike_rate")
graph.add_edge(START, "runs_in_boundary")
graph.add_edge(START, "balls_per_boundary")
graph.add_edge("strike_rate", "summary")
graph.add_edge("runs_in_boundary", "summary")
graph.add_edge("balls_per_boundary", "summary")
graph.add_edge("summary", END)

# Compile the graph
workflow = graph.compile()

# %% Invoke the workflow
response = workflow.invoke({"runs": 143, "balls": 78, "fours": 14, "sixes": 5})
print(response)

# %% Print the graph
display(Image(workflow.get_graph().draw_mermaid_png()))
