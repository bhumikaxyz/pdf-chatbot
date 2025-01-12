# **PDF-QnA Bot: Conversational AI with Document Retrieval**

This project enables users to interact with multiple uploaded PDF documents through a conversational interface. It uses a **Retrieval-Augmented Generation (RAG)** approach, combining document retrieval and natural language generation to answer user queries based on the content of the PDFs.

---

## **Key Features**

* **Document Upload & Processing** : Upload multiple PDF files for processing.
* **Automatic Text Extraction** : The system extracts and processes text from PDF documents.
* **Intelligent Conversational Agent** : Leverages advanced language models to answer questions by retrieving relevant document chunks.
* **Context-Aware Conversations** : The chat history is maintained to ensure relevant and consistent responses.
* **Vector-Based Retrieval** : Uses vector embeddings to search for relevant document segments, enhancing the accuracy of answers.

---

## **How It Works**

1. **Upload Documents** : Users can upload one or more PDF files through the sidebar.
2. **Processing** : When the "Process" button is clicked, the system extracts text from the PDFs and splits it into chunks for efficient retrieval.
3. **Question Answering** : After processing, users can input questions in the chat interface, and the system generates answers based on the document content.
4. **Conversational Memory** : The conversation history is stored, allowing the system to reference past interactions for improved context in ongoing conversations.

---

## **Technologies Used**

* **Streamlit** : Interactive web app framework for the frontend interface.
* **Google Generative AI** : Utilized for text embeddings and natural language generation, powered by the Gemini model.
* **LangChain** : Framework that integrates document retrieval and generative models, enhancing conversational abilities.
* **FAISS** : Vector database used for storing document chunks and performing similarity-based retrieval.
* **RecursiveCharacterTextSplitter** : Divides large text documents into smaller, digestible chunks.
* **ConversationBufferMemory** : Keeps track of the ongoing conversation to ensure contextual understanding.
* **PyPDF2** : Extracts raw text from PDF files.
* **Sentence-Transformers** : Provides semantic embeddings to represent text in vector space for efficient similarity searches.
* **Dotenv** : Loads environment variables for secure handling of API keys.

---

## **Installation and Setup**

1. Clone the repository:

   ```bash
   git clone https://github.com/bhumikaxyz/pdf-chatbot.git
   cd pdf-chatbot
   ```
2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Set up your environment variables in a `.env` file with the following entries:

   ```plaintext
   GOOGLE_API_KEY=<your_google_api_key>
   ```
4. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

---

### **Usage**

1. **Upload Documents** : Go to the sidebar and upload multiple PDF files.
2. **Process Files** : Click on the "Process" button to extract text and prepare the documents for querying.
3. **Ask Questions** : Enter your questions in the chat input field. The system will return answers by referencing the uploaded documents.
4. **Interactive Chat** : As the conversation progresses, the system remembers previous messages, providing relevant and context-aware responses.

---

## **Dependencies**

* `streamlit`
* `PyPDF2`
* `langchain`
* `langchain_community`
* `sentence_transformers`
* `langchain_google_genai`
* `google-generativeai`
* `dotenv`

---

## **Contributing**

Feel free to contribute to this project by submitting issues or pull requests. If you have suggestions for improvements or new features, don’t hesitate to share!

---

## License

This project is licensed under the MIT License.
