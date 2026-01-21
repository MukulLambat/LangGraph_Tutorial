# %% import dependencies
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from IPython.display import Image, display
from langchain_openai import ChatOpenAI


# %% Define state
class qa_state(TypedDict):
    question: str
    answer: str


# %% Define function to response the question
def llm_response(state: qa_state) -> qa_state:
    llm = ChatOpenAI(
        base_url="http://127.0.0.1:8080",
        api_key="not-needed",
        model="local-model",
    )
    prompt = f"answer the question given below, \n\n question: \n{state['question']}"
    response = llm.invoke(prompt)
    state["answer"] = response.content
    return state


# %% Define graph
qa_graph = StateGraph(qa_state)

# %% Define nodes
qa_graph.add_node("llm_answer", llm_response)

# %% Define Edges
qa_graph.add_edge(START, "llm_answer")
qa_graph.add_edge("llm_answer", END)


# %% Compile graph
workflow = qa_graph.compile()
# %%
query = {"question": "Who is the prime minister or India?"}
final_answer = workflow.invoke(query)
display(Image(workflow.get_graph().draw_mermaid_png()))

# %%
print(final_answer)

# %%
