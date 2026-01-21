# %% import dependencies
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from IPython.display import Image, display
from langchain_openai.chat_models import ChatOpenAI

# %% Define LLM
llm = ChatOpenAI(
    base_url="http://127.0.0.1:8080",
    api_key="not needed",
    model="local_model",
)


# %% Define state of the workflow
class blog_state(TypedDict):
    topic: str
    outline: str
    genrated_blog: str


# %% Define function to generate outline on the topic
def outline(state: blog_state) -> blog_state:
    topic = state["topic"]
    prompt = f"Create a outline on the topic {topic} to write blog on it."
    outline = llm.invoke(prompt)
    state["outline"] = outline.content
    return state


# %% Define function to generate blog on the outline of the topic
def blog(state: blog_state) -> blog_state:
    outline = state["outline"]
    prompt = f"Generate a detailed blog on topic given bwelow \ntopic: {state['topic']}\n\n Based on the given below outline, \n\n {outline}"
    blog = llm.invoke(prompt)
    state["genrated_blog"] = blog.content
    return state


# %% Define graph
graph = StateGraph(blog_state)

graph.add_node("generate_outline", outline)
graph.add_node("generate_blog", blog)

graph.add_edge(START, "generate_outline")
graph.add_edge("generate_outline", "generate_blog")
graph.add_edge("generate_blog", END)
# %% Input to the workflow
workflow = graph.compile()
blog_topic = {"topic": "cricket"}
detailed_blog = workflow.invoke(blog_topic)
print(detailed_blog)
display(Image(workflow.get_graph().draw_mermaid_png()))

# %%
print(detailed_blog["genrated_blog"])

# %%
