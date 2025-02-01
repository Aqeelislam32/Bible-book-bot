# Bible-book-bot

**1. Overview**  
Bible Bot is an AI-powered Streamlit application designed to provide insightful, context-aware answers to questions about the Bible. Leveraging Mistral AI for natural language processing and Pinecone for efficient vector-based retrieval, the app delivers accurate, scripture-backed responses. Its user-friendly interface includes a chat-like experience with customizable styling, predefined FAQs, and persistent chat history.

---

**2. Features**  
- **AI-Powered Insights**: Utilizes Mistral AI through LangChain for generating human-like, context-aware responses.  
- **Semantic Search**: Integrates Pinecone vector storage and HuggingFace embeddings for retrieving relevant biblical content.  
- **Interactive UI**: Custom Streamlit interface with chat bubbles, avatars, and a sidebar for quick access to common queries.  
- **Session History**: Maintains chat history within the session for continuity.  
- **Predefined Prompts**: Sidebar includes FAQs like creation stories, covenants, and key figures (e.g., Abraham) for quick exploration.  
- **Caching**: Implements `InMemoryCache` to optimize response speed for repeated queries.  

---

**3. Description**  
**Technical Workflow**:  
1. **Embeddings & Vector Store**: Text data is processed using HuggingFace embeddings and stored in Pinecone for fast similarity searches.  
2. **RetrievalQA Chain**: Combines Mistral AI with Pinecone to fetch and generate answers, ensuring responses are grounded in biblical texts.  
3. **Custom Prompts**: Uses `ChatPromptTemplate` to structure inputs for clarity and relevance.  

**User Experience**:  
- **Welcome Message**: Introduces the bot’s purpose with a friendly greeting and emojis.  
- **Input/Output Design**:  
  - *User Input*: A prominently styled text box with a green border invites questions.  
  - *Responses*: Answers are displayed in distinct chat bubbles—user queries in dark gray and bot responses in lighter gray, each with thematic avatars (e.g., a Bible or dove GIF).  
- **Sidebar Tools**:  
  - Displays book metadata (e.g., Bible version, author).  
  - Offers quick-access questions to guide new users.  
- **History Management**: A "Delete Chat History" button resets conversations, ensuring privacy.  

---

**4. Concise Summary**  
Bible Bot is a Streamlit-based AI assistant that combines Mistral AI’s language capabilities with Pinecone’s vector search to answer biblical questions. Its intuitive design, including chat history, predefined prompts, and visually appealing UI, makes exploring scripture engaging and accessible.aaa
