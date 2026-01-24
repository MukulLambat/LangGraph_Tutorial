# %% Import Dependencies
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from IPython.display import Image, display
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import json
import operator


# %% User essay

user_essay = """India in the Age of AI
As the world enters a transformative era defined by artificial intelligence (AI), India stands at a critical juncture — one where it can either emerge as a global leader in AI innovation or risk falling behind in the technology race. The age of AI brings with it immense promise as well as unprecedented challenges, and how India navigates this landscape will shape its socio-economic and geopolitical future.

India's strengths in the AI domain are rooted in its vast pool of skilled engineers, a thriving IT industry, and a growing startup ecosystem. With over 5 million STEM graduates annually and a burgeoning base of AI researchers, India possesses the intellectual capital required to build cutting-edge AI systems. Institutions like IITs, IIITs, and IISc have begun fostering AI research, while private players such as TCS, Infosys, and Wipro are integrating AI into their global services. In 2020, the government launched the National AI Strategy (AI for All) with a focus on inclusive growth, aiming to leverage AI in healthcare, agriculture, education, and smart mobility.

One of the most promising applications of AI in India lies in agriculture, where predictive analytics can guide farmers on optimal sowing times, weather forecasts, and pest control. In healthcare, AI-powered diagnostics can help address India’s doctor-patient ratio crisis, particularly in rural areas. Educational platforms are increasingly using AI to personalize learning paths, while smart governance tools are helping improve public service delivery and fraud detection.

However, the path to AI-led growth is riddled with challenges. Chief among them is the digital divide. While metropolitan cities may embrace AI-driven solutions, rural India continues to struggle with basic internet access and digital literacy. The risk of job displacement due to automation also looms large, especially for low-skilled workers. Without effective skilling and re-skilling programs, AI could exacerbate existing socio-economic inequalities.

Another pressing concern is data privacy and ethics. As AI systems rely heavily on vast datasets, ensuring that personal data is used transparently and responsibly becomes vital. India is still shaping its data protection laws, and in the absence of a strong regulatory framework, AI systems may risk misuse or bias.

To harness AI responsibly, India must adopt a multi-stakeholder approach involving the government, academia, industry, and civil society. Policies should promote open datasets, encourage responsible innovation, and ensure ethical AI practices. There is also a need for international collaboration, particularly with countries leading in AI research, to gain strategic advantage and ensure interoperability in global systems.

India’s demographic dividend, when paired with responsible AI adoption, can unlock massive economic growth, improve governance, and uplift marginalized communities. But this vision will only materialize if AI is seen not merely as a tool for automation, but as an enabler of human-centered development.

In conclusion, India in the age of AI is a story in the making — one of opportunity, responsibility, and transformation. The decisions we make today will not just determine India’s AI trajectory, but also its future as an inclusive, equitable, and innovation-driven society."""


# %% define llm instance
# Define your schema with Pydantic
class llm_eval_schema(BaseModel):
    text_feedback: str = Field(description="detailed feedback for the essay")
    score: int = Field(description="score for the essay out of 10", ge=0, le=10)


llm = ChatOllama(model="granite3.2:2b")

llm_stru_out = llm.with_structured_output(llm_eval_schema)

# %% Define state of the graph


class eval_state(TypedDict):
    user_essay: str
    clarity_of_thought: str
    depth_of_analysis: str
    language: str
    summary_feedback: str
    individual_score: Annotated[list[int], operator.add]
    avg_score: float


# %% Define node functions
def clarity_thought_fcn(state: eval_state):
    essay = state["user_essay"]
    prompt = f"Evaluate the essay given below for clarity of thoughts and provide score out of 10,\n {essay}"
    result = llm_stru_out.invoke(prompt)
    return {
        "clarity_of_thought": result.text_feedback,
        "individual_score": [result.score],
    }


def depth_analysis_fcn(state: eval_state):
    essay = state["user_essay"]
    prompt = f"Evaluate the essay given below for in depth analysis and provide score out of 10,\n {essay}"
    result = llm_stru_out.invoke(prompt)
    return {
        "depth_of_analysis": result.text_feedback,
        "individual_score": [result.score],
    }


def language_fcn(state: eval_state):
    essay = state["user_essay"]
    prompt = f"Evaluate the essay given below for the language written and provide score out of 10,\n {essay}"
    result = llm_stru_out.invoke(prompt)
    return {
        "language": result.text_feedback,
        "individual_score": [result.score],
    }


def summarized_fcn(state: eval_state):

    # summary feedback
    prompt = f'Based on the following feedbacks create a summarized feedback \n language feedback - {state["language"]} \n depth of analysis feedback - {state["depth_of_analysis"]} \n clarity of thought feedback - {state["clarity_of_thought"]}'

    overall_feedback = llm.invoke(prompt).content

    # avg calculate
    avg_score = sum(state["individual_score"]) / len(state["individual_score"])

    return {"summary_feedback": overall_feedback, "avg_score": avg_score}


# %% Create graph

graph = StateGraph(eval_state)

# Define node for the graph
graph.add_node("evaluate_clarity_of_thought", clarity_thought_fcn)
graph.add_node("evaluate_depth_analysis", depth_analysis_fcn)
graph.add_node("evaluate_language", language_fcn)
graph.add_node("summarized_feedback", summarized_fcn)

# Define edges for the graph
graph.add_edge(START, "evaluate_clarity_of_thought")
graph.add_edge(START, "evaluate_depth_analysis")
graph.add_edge(START, "evaluate_language")
graph.add_edge("evaluate_clarity_of_thought", "summarized_feedback")
graph.add_edge("evaluate_depth_analysis", "summarized_feedback")
graph.add_edge("evaluate_language", "summarized_feedback")
graph.add_edge("summarized_feedback", END)

# compile the graph

workflow = graph.compile()

# %% Invoke the workflow
response = workflow.invoke({"user_essay": user_essay})
print(response)

# %% Print the graph
display(Image(workflow.get_graph().draw_mermaid_png()))
# %%

print(response["avg_score"])
# print(response["individual_score"])
print(response["summary_feedback"])

# %%
