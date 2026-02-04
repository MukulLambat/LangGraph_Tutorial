# %% Import dependencies
from langchain_openai.chat_models import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver

# %% Define LLM instance
llm = ChatOpenAI(base_url="http://127.0.0.1:8080", api_key="not required")
# response = llm.invoke("hello").content
# print(response)


# %% Define chatbot state
class chatstate(TypedDict):
    messages: Annotated[BaseMessage, operator.add]


# %% Define functions for the nodes
def llm_response(state: chatstate):

    response = llm.invoke(state["messages"])

    return {"messages": [response]}


# %% Define graph

# Define checkpointer to provide memory to chatbot
checkpoint = MemorySaver()

chatbot_graph = StateGraph(chatstate)

# Define nodes
chatbot_graph.add_node("llm_response", llm_response)

# Define edges of the graph
chatbot_graph.add_edge(START, "llm_response")
chatbot_graph.add_edge("llm_response", END)

# compile the graph
chatbot = chatbot_graph.compile(checkpointer=checkpoint)

# %% Chatbot like a conversation with history
thread_id = 1

while True:
    user_query = input("Type your query: ")

    if user_query.strip().lower() in ["exit", "bye", "end"]:
        break

    config = {"configurable": {"thread_id": thread_id}}
    response = chatbot.invoke(
        {"messages": [HumanMessage(content=user_query)]}, config=config
    )

    print(response["messages"][-1].content)

# %%
