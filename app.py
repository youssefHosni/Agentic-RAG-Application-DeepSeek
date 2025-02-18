import time 
from crewai import Crew
from tools import create_pdf_tool, web_search_tool
from agents import create_router_agent, create_retriever_agent, create_grader_agent, create_hallucination_grader, create_answer_grader
from tasks import create_router_task, create_retriever_task, create_grader_task, create_hallucination_task, create_answer_task

def main():
    # Download the PDF
    pdf_filename = 'data/attention_is_all_you_need.pdf'
    
    # Create tools
    rag_tool = create_pdf_tool(pdf_filename)

    # Create agents
    router_agent = create_router_agent()
    retriever_agent = create_retriever_agent()
    grader_agent = create_grader_agent()
    hallucination_grader = create_hallucination_grader()
    answer_grader = create_answer_grader()

    # Create tasks
    router_task = create_router_task(router_agent)
    retriever_task = create_retriever_task(retriever_agent, router_task, rag_tool, web_search_tool)
    grader_task = create_grader_task(grader_agent, retriever_task)
    hallucination_task = create_hallucination_task(hallucination_grader, grader_task)
    answer_task = create_answer_task(answer_grader, hallucination_task, web_search_tool)

    # Create crew
    rag_crew = Crew(
        agents=[router_agent, retriever_agent, grader_agent, hallucination_grader, answer_grader],
        tasks=[router_task, retriever_task, grader_task, hallucination_task, answer_task],
        verbose=True,
    )
    start_time = time.time()
    # Kickoff the crew with a user question
    user_question = {"question": "what is the weather in New York city?"}
    result = rag_crew.kickoff(inputs=user_question)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds.")
    print(result)


if __name__ == "__main__":
    main()