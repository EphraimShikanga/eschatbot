"""
Chatbot logic and functionality
"""

import argparse
import os

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate

# from langchain_community.chat_models import ChatGooglePalm
from langchain_community.embeddings import GooglePalmEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

CHROMA_DB_PATH = "database/chroma"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
api_key_anthropic = os.getenv("ANTHROPIC_API_KEY")


def main():
    """
    Main function for the chatbot
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="Query text")
    args = parser.parse_args()
    query_text = args.query_text

    # print(f"Query text: {query_text}")

    # load the database created earlier
    # embedding_function = GoogleGenerativeAIEmbeddings(
    #     model="models/embedding-001", google_api_key=api_key
    # )
    embedding_function = GooglePalmEmbeddings(google_api_key=api_key)
    db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_function)

    results = db.similarity_search_with_relevance_scores(query_text, k=3)

    if len(results) == 0 or results[0][1] < 0.5:
        print("No results found")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(f"Prompt: {prompt}")
    # print(f"Context text: {context_text}")

    model_one = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)
    # model_two = ChatGooglePalm()

    response_one = model_one.invoke(prompt)
    # response_two = model_two.invoke(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response_one = f"Response: {response_one}\nSources: {sources}"
    # formatted_response_two = f"Response: {response_two}\nSources: {sources}"
    print(formatted_response_one)
    print("\n")
    # formatted_response_twoo = formatted_response_two.replace("\n", "\n")
    # print(formatted_response_twoo)


if __name__ == "__main__":
    main()
