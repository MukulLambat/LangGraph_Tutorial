# %% Import dependencies
from typing import TypedDict
from langchain_openai.chat_models import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import InMemorySaver

# %% Create a LLM instance
llm = ChatOpenAI(base_url="http://127.0.0.1:8080", api_key="Not Required")
# response = llm.invoke("hello")
# print(response.content)


# %% Create state of the graph
class wokrflow_state(TypedDict):
    topic: str
    gen_joke: str
    joke_exp: str


# %% Create the functions for the graph nodes
def joke_generation_fcn(state: wokrflow_state):
    prompt = f"generate a funny joke on the topic given below,\n {state['topic']}"
    gen_joke = llm.invoke(prompt).content
    return {"gen_joke": gen_joke}


def joke_explanation_fcn(state: wokrflow_state):
    prompt = f"write a explanation of the joke given below,\n {state['gen_joke']}"
    joke_exp = llm.invoke(prompt).content
    return {"joke_exp": joke_exp}


# %% graph of the workflow
joke_explanation_graph = StateGraph(wokrflow_state)

# Create the nodes of the graphs
joke_explanation_graph.add_node("joke_generation", joke_generation_fcn)
joke_explanation_graph.add_node("joke_explanation", joke_explanation_fcn)

# Create the edges of the graph
joke_explanation_graph.add_edge(START, "joke_generation")
joke_explanation_graph.add_edge("joke_generation", "joke_explanation")
joke_explanation_graph.add_edge("joke_explanation", END)

# Create the checkpointer for Persistence memory

checkpointer = InMemorySaver()

workflow = joke_explanation_graph.compile(checkpointer=checkpointer)
# %% invoke the workflow
config1 = {"configurable": {"thread_id": 1}}
joke_explanation = workflow.invoke({"topic": "Rahul Gandhi"}, config=config1)
print(joke_explanation)

# %% To get the state values from memory
final_state = workflow.get_state()
print(final_state)
each_superstep_state = workflow.get_state_history()
print(each_superstep_state)
