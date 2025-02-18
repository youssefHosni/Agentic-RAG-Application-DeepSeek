from crewai import Task

def create_router_task(router_agent):
    return Task(
        description=(
            "Analyze the given question {question} to determine the appropriate search method:\n"
            "\n"
            "1. Use 'vectorstore' if:\n"
            "   - The question contains a keyword or a similar words\n"
            "   - The topic is likely covered in our vector database\n"
            "\n"
            "2. Use 'web_search' if:\n"
            "   - The topic requires current or real-time information\n"
            "   - The question is about general topics not covered in our vector database\n"
            "\n"
            "Make decisions based on semantic understanding rather than keyword matching."
        ),
        expected_output=(
            "Return exactly one word:\n"
            "'vectorstore' - if the question can be answered from our RAG knowledge base\n"
            "'web_search' - if the question requires external information\n"
            "No additional explanation or preamble should be included."
        ),
        agent=router_agent,
    )


def create_retriever_task(retriever_agent, router_task, rag_tool, web_search_tool):
    return Task(
        description=(
            "Based on the response from the router task extract information for the question {question} with the help of the respective tool."
            "Use the web_serach_tool to retrieve information from the web in case the router task output is 'websearch'. You should pass the input query {question} to the web_search_tool." 
            "Use the rag_tool to retrieve information from the vectorstore in case the router task output is 'vectorstore'."
        ),
        expected_output=(
            "You should analyse the output of the 'router_task'"
            "If the response is 'websearch' then use the web_search_tool to retrieve information from the web."
            "If the response is 'vectorstore' then use the rag_tool to retrieve information from the vectorstore."
            "Return a claer and consise text as response."
        ),
        agent=retriever_agent,
        context=[router_task],
        tools=[rag_tool, web_search_tool],
    )


def create_grader_task(grader_agent, retriever_task):
    return Task(
        description=(
            "Based on the response from the retriever task for the quetion {question} evaluate whether the retrieved content is relevant to the question."
        ),
        expected_output=(
            "Binary score 'yes' or 'no' score to indicate whether the document is relevant to the question"
            "You must answer 'yes' if the response from the 'retriever_task' is in alignment with the question asked."
            "You must answer 'no' if the response from the 'retriever_task' is not in alignment with the question asked."
            "Do not provide any preamble or explanations except for 'yes' or 'no'."
        ),
        agent=grader_agent,
        context=[retriever_task],
    )


def create_hallucination_task(hallucination_grader, grader_task):
    return Task(
        description=(
            "Based on the response from the grader task for the quetion {question} evaluate whether the answer is grounded in / supported by a set of facts."
        ),
        expected_output=(
            "Binary score 'yes' or 'no' score to indicate whether the answer is sync with the question asked"
            "Respond 'yes' if the answer is in useful and contains fact about the question asked."
            "Respond 'no' if the answer is not useful and does not contains fact about the question asked."
            "Do not provide any preamble or explanations except for 'yes' or 'no'."
        ),
        agent=hallucination_grader,
        context=[grader_task],
    )


def create_answer_task(answer_grader, hallucination_task, web_search_tool):
    return Task(
        description=(
            "Based on the response from the hallucination task for the quetion {question} evaluate whether the answer is useful to resolve the question."
            "If the answer is 'yes' return a clear and concise answer."
            "If the answer is 'no' then perform a 'websearch' and return the response"
        ),
        expected_output=(
            "Return a clear and concise response if the response from 'hallucination_task' is 'yes'."
            "Perform a web search using 'web_search_tool' and return ta clear and concise response only if the response from 'hallucination_task' is 'no'."
            "Otherwise respond as 'Sorry! unable to find a valid response'."
            "Make sure the final response is clear and concise and contain only the answer to the input question without any preamble or explanation as this answer will be presnted to the user."
            "The final answer should be a clear and concise response to the input question."
        ),
        context=[hallucination_task],
        agent=answer_grader,
        tool = [web_search_tool],   
        
    )