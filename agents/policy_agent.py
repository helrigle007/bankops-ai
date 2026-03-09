"""
RAG-grounded policy agent for banking procedure and compliance questions.

Retrieves relevant policy documents from the vectorstore and generates
grounded answers with source citations. Refuses to answer without
supporting evidence to prevent hallucination.
"""

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from knowledge.ingest import get_retriever


POLICY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a policy compliance assistant for a commercial banking operations team.

Your role is to answer questions about banking procedures, policies, and regulatory requirements
using ONLY the provided context from official policy documents.

Rules:
1. ONLY answer based on the provided context — never use general knowledge
2. If the context doesn't contain enough information, say so clearly
3. Cite the specific policy section (document name + section header) for every claim
4. If a question involves a specific dollar amount or threshold, quote the exact policy numbers
5. Flag any compliance-sensitive items that may need Risk or Legal review
6. Use clear, direct language — no hedging unless the policy itself is ambiguous

Context from policy documents:
{context}
"""),
    ("human", "{question}"),
])


def format_docs(docs):
    """Format retrieved documents with source metadata."""
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        # Extract just the filename from the path
        source_name = source.split("/")[-1].replace(".md", "").replace("_", " ").title()
        formatted.append(f"[Source: {source_name}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def get_policy_chain(model: str = "claude-sonnet-4-6"):
    """
    Build a RAG chain that retrieves policy docs and generates grounded answers.
    """
    retriever = get_retriever(k=4)
    llm = ChatAnthropic(model=model, temperature=0)

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | POLICY_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain


def query_policy(question: str, model: str = "claude-sonnet-4-6") -> dict:
    """
    Query the policy knowledge base and return the answer with retrieval metadata.
    """
    retriever = get_retriever(k=4)
    llm = ChatAnthropic(model=model, temperature=0)

    # Retrieve docs separately so we can return them as metadata
    docs = retriever.invoke(question)
    context = format_docs(docs)

    chain = POLICY_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    sources = list(set(
        doc.metadata.get("source", "Unknown").split("/")[-1].replace(".md", "").replace("_", " ").title()
        for doc in docs
    ))

    return {
        "answer": answer,
        "sources": sources,
        "chunks_retrieved": len(docs),
    }


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    questions = [
        "What authorization is required for a $75,000 domestic wire transfer?",
        "How long do we have to resolve a Reg E dispute?",
        "What documentation is needed to add an authorized signer to a commercial account?",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        result = query_policy(q)
        print(f"A: {result['answer'][:200]}...")
        print(f"Sources: {', '.join(result['sources'])}")
