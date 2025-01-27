import os
import torch
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_mistralai import ChatMistralAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain.prompts import PromptTemplate
from PIL import Image

# Initialize Streamlit App Customization using CSS
st.markdown("""
    <style>
        /* Set the main background color */
        .main {
            background-color: #f0f2f6;
        }
        /* Customize the appearance of Streamlit buttons */
        div.stButton > button {
            background-color: #4CAF50; /* Green background */
            color: white; /* White text */
            border-radius: 5px; /* Rounded corners */
            border: 1px solid #4CAF50; /* Green border */
        }
        /* Style for chat messages */
        .chat-message {
            padding: 1.5rem; 
            border-radius: 0.5rem; 
            margin-bottom: 1rem; 
            display: flex;
        }
        /* User message styling */
        .chat-message.user {
            background-color: #2b313e; /* Dark background for user */
        }
        /* Bot message styling */
        .chat-message.bot {
            background-color: #475063; /* Slightly lighter background for bot */
        }
        /* Avatar container */
        .chat-message .avatar {
            width: 20%; /* Allocate 20% width for avatar */
        }
        /* Avatar image styling */
        .chat-message .avatar img {
            max-width: 50px; /* Maximum width */
            max-height: 50px; /* Maximum height */
            border-radius: 50%; /* Circular avatar */
            object-fit: cover; /* Ensure image covers the container */
        }
        /* Message text styling */
        .chat-message .message {
            width: 80%; /* Allocate 80% width for message */
            padding: 0 1.5rem; /* Horizontal padding */
            color: #fff; /* White text color */
        }

        /* Sidebar customization */
        [data-testid="stSidebar"] .stSelectbox {
            background-color: skyblue !important; /* Light blue background */
            border-radius: 5px; /* Rounded corners */
            padding: 5px; /* Padding inside the select box */
        }

        /* Text input field styling */
        .stTextInput>div>div>input {
            border: 2px solid #4CAF50; /* Green border */
            border-radius: 10px; /* Rounded corners */
            padding: 10px; /* Padding inside the input */
            font-size: 18px; /* Larger font size for readability */
            width: 100%; /* Full width */
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); /* Subtle shadow for depth */
        }
        /* Text input label styling */
        .stTextInput label {
            font-size: 20px; /* Larger label font */
            font-weight: bold; /* Bold label text */
            color: #333; /* Dark label color for contrast */
        }

        /* Header container styling */
        .header-container {
            display: flex; /* Flex layout */
            align-items: center; /* Vertically center items */
            gap: 20px; /* Space between items */
            padding: 20px; /* Padding around the container */
        }

        /* Header text styling */
        .header-container h1 {
            font-size: 32px; /* Larger font size */
            color: #222; /* Dark text color */
            margin: 0; /* Remove default margin */
            font-family: 'Arial', sans-serif; /* Modern font */
            line-height: 1.2; /* Tight line spacing */
        }

        /* Set base theme and background color */
        body {
            background-color: #d2c5c5; /* Light background color */
            color: #000; /* Black text for contrast */
        }

        /* Customize the Streamlit container */
        .stApp {
            background-color: #d2c5c5; /* Matching background color */
        }

        /* Customize sidebar content background */
        .sidebar .sidebar-content {
            background-color: #d2c5c5; /* Light background for sidebar */
        }
    </style>
""", unsafe_allow_html=True)

# *Hardcoded Environment Variables*
# These are the API keys and endpoints required for the bot to function.
# *Security Note:* Hardcoding API keys is not recommended for production.
# Consider using environment variables or secret managers instead.



# *Set up environment variables*
# These lines set the environment variables for use in the application.
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT")
os.environ["MISTRAL_API_KEY"] = os.getenv("MISTRAL_API_KEY")
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")

# *Initialize the LLM (Language Model) with the system prompt in Serbian*
# The system prompt defines the behavior and structure of the bot's responses.
system_prompt = """ *System Prompt for BibleBot*  

You are *BibleBot*, a highly knowledgeable, professional, and user-friendly assistant designed to provide accurate, insightful, and contextually rich answers related to the Bible. Your role is to assist users—including scholars, clergy, and the general public—with questions about biblical texts, interpretations, and teachings. Always follow this structure when generating answers:  

---

### *Response Structure*  

1. **Understanding the Query and Relevant Context**:  
   - Carefully analyze the user's query.  
   - Utilize provided scripture passages or biblical references to craft precise, contextually relevant answers.  

2. **Structure for Every Response**:  
   For each response, adhere to the following structure:  

   - **Clear Summary (Immediate Value)**: Start with a short, clear summary (1-2 sentences) that directly addresses the user's question. Avoid complex theological jargon; use clear and professional language.  

   - **Relevant Scripture or Biblical Reference**: Cite specific verses, chapters, or passages that support your response. Ensure accuracy and include proper references for increased reliability.  

   - **Detailed Explanation**: Expand on the summary with a detailed explanation of the biblical text, including "why" and "how" it answers the query. Address nuances, interpretations, or potential misunderstandings.  

   - **Practical Application/Insights**: Provide practical advice or insights on how the user can apply or reflect on the biblical teaching in their personal or spiritual life, if applicable.  

   - **Additional Context or Interpretations (Optional)**: Highlight alternative interpretations, historical context, or theological perspectives, ensuring a balanced and comprehensive response.  

   - **Example/Scenario (Optional, but Effective)**: Where applicable, include real-life examples or hypothetical scenarios to make the explanation more relatable and understandable.  

   - **Resources and References**: Guide the user to additional resources, such as online Bible tools, commentaries, or theological texts for deeper study.  

   - **Follow-Up Suggestions or Support**: For complex queries or deeper insights, recommend reaching out to local clergy, biblical scholars, or online Bible study groups.  

3. **Tone and Style**:  
   - Be professional yet approachable.  
   - Avoid overly complex or verbose answers.  
   - Use concise sentences and bullet points for clarity, where applicable.  

4. **Handling Unanswered Queries**:  
   - If the query cannot be answered directly, kindly explain why and offer alternative resources or steps the user can take.  

5. **Example Response for Illustration**:  
   *Query*: “What does the Bible say about forgiveness?”  
   *Response*:  
   - **Summary**: “The Bible emphasizes forgiveness as a cornerstone of Christian faith, urging believers to forgive others as God has forgiven them.”  
   - **Relevant Scripture**: “Ephesians 4:32 - ‘Be kind to one another, tenderhearted, forgiving one another, as God in Christ forgave you.’”  
   - **Explanation**: “Forgiveness is central to Jesus' teachings. It reflects God’s grace and allows for reconciliation. Jesus illustrated this in the Parable of the Unforgiving Servant (Matthew 18:21-35).”  
   - **Practical Application**:  
     1. Reflect on areas in life where forgiveness is needed.  
     2. Pray for the strength to forgive, following Jesus' example.  
   - **Example**: “Consider someone holding a grudge against a friend. By forgiving, they reflect Christ’s love and free themselves from bitterness.”  
   - **Resources**: “Explore commentaries on forgiveness, such as those available on BibleGateway or GotQuestions.org.”  
   - **Follow-Up**: “If you seek spiritual guidance, consider speaking to a pastor or joining a Bible study group for deeper discussion.”  

---

### *Why This Works*  
- **Clarity**: The summary provides an immediate answer.  
- **Authority**: Scriptural references establish trust.  
- **Practicality**: Applications make it actionable.  
- **Depth**: Context and examples reduce ambiguity.  
- **Support**: Follow-up suggestions ensure user satisfaction for complex queries.  

Using this structure, *BibleBot* will deliver authoritative, personalized, and comprehensive answers to all Bible-related questions in English.  """


# Initialize the Language Model (LLM) with the system prompt
llm = ChatMistralAI(model="mistral-large-latest", system_message=system_prompt)

# *Initialize Pinecone for Vector Database*
# Initialize Pinecone for vector database
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = "us-east-1"  # Ensure this matches your Pinecone environment
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

# *Connect to Pinecone Index*
# Connect to the specific Pinecone index that contains regulations data.
index_name = "data"  # Name of the Pinecone index for regulations
index = pc.Index(index_name)

# *Initialize Embedding Model*
# HuggingFaceEmbeddings converts text into vectors for similarity search.
embedding_function = HuggingFaceEmbeddings(
    model_name='BAAI/bge-small-en-v1.5',  # Specify the embedding model
    model_kwargs={
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available
    }
)

# *Check for CUDA Availability*
# Inform the user if the model is running on CPU, which may be slower.
if not torch.cuda.is_available():
    st.warning("Warning: CUDA is not available. The model will run on CPU, which may lead to slower performance.")

# *Create Pinecone VectorStore*
# This integrates Pinecone with LangChain for vector-based retrieval.
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embedding_function,
    text_key='text',  # The key in your Pinecone index where the text is stored
    namespace="text_chunks"  # Namespace for organizing vectors
)

# *Initialize Retriever*
# The retriever fetches the top 'k' relevant text chunks for a given query.
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})  # Retrieve top 4 relevant chunks

# *Define the Query Refinement Prompt Template in Serbian*
# This prompt refines user queries to improve retrieval accuracy.
refinement_template = """Create a focused Bible search query for the RAG retriever bot. Convert to English if it's not already. Include key terms, synonyms, and scripture-specific terminology. Remove unnecessary words. The output should be only the refined query in the following format: {{refined_query}},{{keyterms}},{{synonyms}}

Query: {original_question}

Refined Query:"""


# *Create a PromptTemplate for Query Refinement*
# This template structures how the query is refined before retrieval.
refinement_prompt = PromptTemplate(
    input_variables=["original_question"],  # Define the input variable
    template=refinement_template  # Use the defined refinement template
)

# *Create an LLMChain for Query Refinement*
# Combines the refinement prompt with the language model to process queries.
refinement_chain = refinement_prompt | llm

# *Combine the System Prompt with the Retrieval Prompt Template in Serbian*
# This template structures how the bot uses the retrieved context to answer questions.
combined_template = f"""{system_prompt}

Please answer the following question using only the provided context:  
{{context}}

Question: {{question}}  
Answer:"""


# *Create a ChatPromptTemplate from the Combined Template*
# This prepares the prompt for the retrieval chain.
retrieval_prompt = ChatPromptTemplate.from_template(combined_template)

# *Create a Retrieval Chain with the Combined Prompt*
# The RetrievalQA chain uses the language model and retriever to generate answers.
retrieval_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # Specifies the type of chain; 'stuff' aggregates all documents
    retriever=retriever,
    chain_type_kwargs={"prompt": retrieval_prompt}  # Pass the combined prompt
)

def process_query(query: str) -> str:
    """
    Process a single query and return the bot's response.
    
    Args:
        query (str): The user's question.
    
    Returns:
        str: The bot's structured answer.
    """
    try:
        # *Refine the Query*
        # The user's original question is refined to improve retrieval relevance.
        refined_query_msg = refinement_chain.invoke({"original_question": query})
        
        # *Extract the Refined Query*
        if isinstance(refined_query_msg, dict):
            refined_query = refined_query_msg.get("text", "").strip()
        elif hasattr(refined_query_msg, 'content'):
            refined_query = refined_query_msg.content.strip()
        else:
            refined_query = str(refined_query_msg).strip()

        # *Use the Refined Query in the Retrieval Chain*
        response_msg = retrieval_chain.invoke(refined_query)

        # *Extract the Response from the Retrieval Chain*
        if isinstance(response_msg, dict):
            response = response_msg.get("result", "")
        elif hasattr(response_msg, 'content'):
            response = response_msg.content
        else:
            response = str(response_msg)
        
        return response
    except Exception as e:
        # *Error Handling*
        return f"An error has occurred: {str(e)}"

# *Streamlit App Interface*

# *Display Header*
# This creates a styled header for the chatbot interface.
st.markdown("""
    <h1 style="text-shadow: 2px 2px 5px #4CAF50; font-weight: bold; text-align: center;">
         📚Bible Chatbot📒📝
    </h1>
""", unsafe_allow_html=True)

# *Display Introduction Text*
# Provides a welcoming message and a brief description of the chatbot.
st.markdown("""
    <p style="font-size: 18px; color: #000000; line-height: 1.6; text-align: center;">
       👋 Welcome to <strong>Bible Bot</strong>, your trusted guide to all things related to the Bible and its teachings. <br> 💡 I am here to assist you with any questions or provide insights into 🕊️ biblical texts and more.
    </p>
""", unsafe_allow_html=True)

# *Sidebar with Common Queries*
st.sidebar.header("Book Details")
st.sidebar.write("**Name:** 📖 The Holy Bible")
st.sidebar.write("**✍️ Author:** English Standard Version Translation Committee")
 
# Displays a list of common questions in the sidebar to guide users.
st.sidebar.title(" 🔍 Frequently Asked Questions")

# *Define Common Queries*
prompts = [
    "1. What is the purpose of the creation story in Genesis?",
    "2. What is the significance of God’s covenant with Noah?",
    "3. How does the Bible describe the relationship between God and humanity??",
    "4. Why is Abraham considered a key figure in the Bible?",
]

# *Display Each Common Query in the Sidebar*
for prompt in prompts:
    st.sidebar.write(prompt)

# *Session State to Save Chat History*
# Streamlit's session state is used to maintain the chat history across user interactions.
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Initialize empty chat history

# *Input Form*
# Allows the user to input their question and submit it.
query = st.text_input("Your Question:")  # Text input field for user's question

# *Submit Button*
# When clicked, processes the user's query and appends the response to chat history.
if st.button("Send") and query:
    response = process_query(query)  # Process the user's query
    st.session_state.chat_history.append({"question": query, "answer": response})  # Save to chat history

# *Display Chat History*
# Iterates over the chat history and displays each user and bot message.
for entry in st.session_state.chat_history:
    # *Display User Message*
    st.markdown(f'''
        <div class="chat-message user">
            <div class="avatar">
                <img src="https://th.bing.com/th/id/OIP.uDqZFTOXkEWF9PPDHLCntAHaHa?pid=ImgDet&rs=1">
            </div>
            <div class="message">{entry["question"]}</div>
        </div>
    ''', unsafe_allow_html=True)
    
    # *Display Bot Response*
    st.markdown(f'''
        <div class="chat-message bot">
            <div class="avatar">
                <img src="https://i.pinimg.com/originals/0c/67/5a/0c675a8e1061478d2b7b21b330093444.gif" style="max-height: 70px; max-width: 50px;">
            </div>
            <div class="message">{entry["answer"]}</div>
        </div>
    ''', unsafe_allow_html=True)
    
    # *Separator Between Messages*
    st.write("---")  # Horizontal line for separation

# *Option to Clear Chat History*
# Provides a button to reset the chat history for a fresh start.
if st.button("Delete Chat History"):
    st.session_state.chat_history = []  # Clear chat history
    st.experimental_rerun()  # Rerun the app to update the UI
