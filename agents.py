from crewai import Agent 
from crewai import LLM
from config import OPEN_ROUTER_API_KEY

#llm = LLM(model="deepseek-r1:1.5b", base_url="http://localhost:11434")

llm = LLM(model="openrouter/deepseek/deepseek-r1",
          temperature=0,
          api_key=OPEN_ROUTER_API_KEY
          )

def create_router_agent():
    return Agent(
        role='Router',
        goal='Route user questions to either vectorstore or web search based on content relevance',
        backstory=(
            "You are an expert at determining whether a question can be answered using the "
            "information stored in our vector database, or requires a web search. "
            "You understand that the vector database contains comprehensive knowledge base "
            "You make routing decisions based on the semantic meaning of questions rather than just keyword matching."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )


def create_retriever_agent():
    return Agent(
        role="Retriever",
        goal="Use the information retrieved from the vectorstore to answer the question",
        backstory=(
            "You are an assistant for question-answering tasks."
            "Use the information present in the retrieved context to answer the question."
            "You have to provide a clear concise answer."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )


def create_grader_agent():
    return Agent(
        role='Answer Grader',
        goal='Filter out erroneous retrievals',
        backstory=(
            "You are a grader assessing relevance of a retrieved document to a user question."
            "If the document contains keywords related to the user question, grade it as relevant."
            "It does not need to be a stringent test.You have to make sure that the answer is relevant to the question."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )
    
    
def create_hallucination_grader():
    return Agent(
        role="Hallucination Grader",
        goal="Filter out hallucination",
        backstory=(
            "You are a hallucination grader assessing whether an answer is grounded in / supported by a set of facts."
            "Make sure you meticulously review the answer and check if the response provided is in alignmnet with the question asked"
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )
 

def create_answer_grader():
    return Agent(
        role="Answer Grader",
        goal="Filter out hallucination from the answer.",
        backstory=(
            "You are a grader assessing whether an answer is useful to resolve a question."
            "Make sure you meticulously review the answer and check if it makes sense for the question asked"
            "If the answer is relevant generate a clear and concise response."
            "If the answer gnerated is not relevant then perform a websearch using 'web_search_tool'"
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )  
 