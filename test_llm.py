from langserve import RemoteRunnable
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from ollama import embeddings

# Server configuration
OLLAMA_BASE_URL = "http://localhost:11434"
MODEL = "mistral"
MODEL_EMBEDDING = "nomic-embed-text"

# Connect to remote LLM server
llm = ChatOllama(model=MODEL, base_url=OLLAMA_BASE_URL)

def single_turn(user_message: str) -> str:
    """Single turn: Send one message and receive a response."""
    response = llm.invoke(user_message)
    return response.content


def multi_turn(assistant_messages: list[str]=None):
    """Multi turn: Maintain conversation history and chat."""
    print(f"Chatbot started (Server: {OLLAMA_BASE_URL}) | Exit: 'quit' or 'exit'")
    history = []

    N = len(assistant_messages) if assistant_messages else int(1e5)  # Default to 2 turns if no messages provided
    
    for i in range(N):
        if not assistant_messages:
            user_input = input("\nUser: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["quit", "exit"]:
                print("Exiting.")
                break
        else: 
            user_input = assistant_messages[i]
            print(f"\nUser: {user_input}")

        history.append(HumanMessage(content=user_input))

        # Invoke remote LLM with conversation history
        response = llm.invoke(history)
        assistant_message = response.content

        print(f"Assistant: {assistant_message}")
        history.append(AIMessage(content=assistant_message))

def find_closest_sentence(query: str, sentences: list[str]) -> str:
    """Find the closest sentence to the query using embedding vectors."""
    
    query_embedding = embeddings(model=MODEL_EMBEDDING, prompt=query)["embedding"]
    
    max_similarity = -1
    closest_sentence = ""
    
    for sentence in sentences:
        sentence_embedding = embeddings(model=MODEL_EMBEDDING, prompt=sentence)["embedding"]
        
        # Calculate cosine similarity
        dot_product = sum(a * b for a, b in zip(query_embedding, sentence_embedding))
        magnitude_query = sum(a ** 2 for a in query_embedding) ** 0.5
        magnitude_sentence = sum(b ** 2 for b in sentence_embedding) ** 0.5
        similarity = dot_product / (magnitude_query * magnitude_sentence)
        
        print(f"  {sentence}: {similarity:.4f}")
        
        if similarity > max_similarity:
            max_similarity = similarity
            closest_sentence = sentence
    
    return closest_sentence


def test_embedding(query):
    """Test embedding-based sentence search."""
    sentences = [
        "The weather is really nice today",
        "Do you know about cats?",
        "Do you know about dogs?",
        "Dogs are animals with high loyalty",
        "Python is a programming language",
        "Cats have an independent personality",
    ]
    
    result = find_closest_sentence(query, sentences)
    print(f"Query: {query}")
    print(f"Closest: {result}\n")

            
if __name__ == "__main__":
    # Single turn test
    print("=== Single Turn Test ===")
    result = single_turn("Hello! Introduce yourself briefly.")

    print(f"Response: {result}\n")

    # Multi turn chat
    print("=== Multi Turn Chat ===")
    multi_turn(['My name is Alice', 'What is my name?'])

    # Embedding vector test
    print("=== Embedding Vector Test ===")
    test_embedding("Tell me about dogs")