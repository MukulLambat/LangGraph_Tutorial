# %% Import dependencies
from langchain_openai.chat_models import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Literal
import operator
from pydantic import BaseModel, Field

# %% Define LLM instance
llm = ChatOpenAI(
    model="granite3.2:2b ", base_url="http://127.0.0.1:8080", api_key="not required"
)


class evalution_schema(BaseModel):

    eval_result: Annotated[
        Literal["approved", "not approved"],
        Field(
            description="Provide whether the generated tweet is good to be posted or not"
        ),
    ]


# Define structured output LLM for LLM evaluation result
structured_llm = llm.with_structured_output(schema=evalution_schema)
# response = structured_llm.invoke("hello")
# print(response)


# %% Define the state for the graph
class Poststate(TypedDict):
    topic: Annotated[str, "Topic for the tweet"]
    initial_gen_tweet: Annotated[str, "Initial generated tweet"]
    feedback: Annotated[str, "feedback on the generated tweet"]
    eval_result: Annotated[
        Literal["approved", "not approved"],
        "Provide whether the generated tweet is good to be posted or not",
    ]
    optimized_tweets: Annotated[list[str], operator.add]
    final_tweet: Annotated[str, "Final tweet after optimization"]
    iteration: int
    max_iteration: int


# %% Define the functions for the nodes
def generate_tweet_fcn(state: Poststate):
    topic = state["topic"]
    prompt = f"Based on the given below topic\n, {topic}\n\n Generate a tweet in less than 80 words."
    initial_gen_tweet = llm.invoke(prompt).content
    return {"initial_gen_tweet": initial_gen_tweet}


def evaluate_tweet_fcn(state: Poststate):
    initial_gen_tweet = state["initial_gen_tweet"]
    prompt = f"Evaluate the tweet given below on the basis of language, content, and user interaction and provide the feedback which can then be used for optimization of tweet.\n\n Tweet:\n {initial_gen_tweet}"
    feedback = llm.invoke(prompt).content
    eval_result = structured_llm.invoke(prompt)
    iteration = state["iteration"] + 1
    return {"feedback": feedback, "eval_result": eval_result, "iteration": iteration}


def optimize_tweet_fcn(state: Poststate):
    initial_gen_tweet = state["initial_gen_tweet"]
    prompt = f"optimize the tweet given,\n\n Tweet:\n {initial_gen_tweet} based of the feedback given below,\n\n Feedback:\n {state['feedback']}"
    optimized_tweet = llm.invoke(prompt).content
    return {"optimized_tweets": [optimized_tweet], "initial_gen_tweet": optimized_tweet}


def get_approval(state: Poststate):
    eval_result = state["eval_result"]
    if eval_result == "approved" or state["iteration"] >= state["max_iteration"]:
        return "approved"
    else:
        return "not approved"


# %% Define the graph
graph = StateGraph(Poststate)

# Define the nodes of the graph
graph.add_node("generate_tweet", generate_tweet_fcn)
graph.add_node("evaluate_tweet", evaluate_tweet_fcn)
graph.add_node("regenerate_tweet", optimize_tweet_fcn)

# Define the edges of the graph
graph.add_edge(START, "generate_tweet")
graph.add_edge("generate_tweet", "evaluate_tweet")
graph.add_conditional_edges(
    "evaluate_tweet",
    get_approval,
    {"approved": END, "not approved": "regenerate_tweet"},
)
graph.add_edge("regenerate_tweet", "evaluate_tweet")

# Compile the graph
workflow = graph.compile()

# %% Invoke the workflow with input
Input = {"topic": "DRS in cricket", "iteration": 0, "max_iteration": 3}
tweet = workflow.invoke(Input)
print(tweet)

# %%
