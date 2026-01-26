# %% Import Dependencies
from langchain_ollama.chat_models import ChatOllama
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# %% Define LLM instance
llm = ChatOllama(model="granite3.2:2b")


class sentiment(BaseModel):
    sentiment: str = Field(
        description="sentiment of the review as Positive or Negative"
    )


stct_llm = llm.with_structured_output(schema=sentiment)


# %% Define state for the graph
class review_state(TypedDict):
    user_review: str
    sentiment: str
    positive_review_response: str
    negative_review_report: str
    negative_review_response: str


# %% Define functions for the graph nodes
def review_sentiment_fcn(state: review_state):
    review = state["user_review"]
    sentiment_obj = stct_llm.invoke(review)
    return {"sentiment": sentiment_obj.sentiment}


def positive_review_response_fcn(state: review_state):

    return {
        "positive_review_response": "Thank you for your response and enjoying the services."
    }


def neg_review_report_fcn(state: review_state):
    prompt = f"Based on the review given by user generate the report of the review given below, \n Review:\n {state['user_review']}"
    report = llm.invoke(prompt)
    return {"negative_review_report": report.content}


def neg_review_response_fcn(state: review_state):
    prompt = f"Based on the report given below of the review, generate a response to show to the user \n Report:\n {state['negative_review_report']}"
    response = llm.invoke(prompt)
    return {"negative_review_response": response.content}


def condition_check(state: review_state):
    sentiment = state["sentiment"].strip().lower()
    if sentiment == "positive":
        return "positive_review_response"
    else:
        return "neg_review_report"


# %% Create the graph

graph = StateGraph(review_state)

# Create the nodes of the graph
graph.add_node("review_sentiment", review_sentiment_fcn)
graph.add_node("positive_review_response", positive_review_response_fcn)
graph.add_node("neg_review_report", neg_review_report_fcn)
graph.add_node("neg_review_response", neg_review_response_fcn)

# Create the edges of the graph
graph.add_edge(START, "review_sentiment")
graph.add_conditional_edges("review_sentiment", condition_check)
graph.add_edge("positive_review_response", END)
graph.add_edge("neg_review_report", "neg_review_response")
graph.add_edge("neg_review_response", END)

# compile the graph

workflow = graph.compile()
# %% Invoke the workflow
review = "Ordered a waffle to find a fingernail and a hair on my food, absolutely shocking customer service as i was told I would have to wait another hour to get a fresh one. I asked for a refund but never received one wouldn't send my worst enemy here"

response = workflow.invoke({"user_review": review})
# print(response)
# %%
print(response["negative_review_response"])
# %%
