from langchain_community.vectorstores import Chroma
from src.helper import load_embedding
from dotenv import load_dotenv
import os
from src.helper import extract_text, validate_url, clear_previous_data
from flask import Flask, render_template, jsonify, request
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from src.helper import load_embedding

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

app = Flask(__name__)

load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embeddings = load_embedding()
persist_directory = "db"
# Now we can load the persisted database from disk, and use it as normal.
vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embeddings)

def qa(input):
    retriever = vectordb.as_retriever()
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    ### Contextualize question ###
    contextualize_q_system_prompt = """
        You are an advanced insurance bot, developed by Legal and General Insurance provider. Your primary role is to deliver highly relevant, accurate, and useful answers to users based on their Query and the available context.
        Please follow these guidelines strictly:
        1. Provide responses directly related to the user's Query. If the query is unclear or insufficient, summarize the Context and include any pertinent details about the Query.
        2. Don't ask the user questions as they don't have the capability to respond.
        3. Strive to provide the best possible results for each Query, like a dedicated insurance chatbot.
        4. Use the Context provided to craft comprehensive, succinct, and user-friendly answers to the Query.
        5. Refer to results from the Context using [context-id] notation for citation. For example: 'some text [1] some other text [2]'.
        6. In cases where the Query relates to multiple subjects sharing the same name, formulate separate responses for each subject to ensure clarity.
        7. Limit responses to a maximum of 300 words to provide concise and focused answers.
        Remember, your ultimate goal is to assist users in navigating information quickly and accurately, in line with the mission of Legal and General.
        """
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_system_prompt = """You are a helpful assistant for answering the queries related to information related to insurance domain. \
        Use the following pieces of retrieved context to answer the question. \
        If you don't know the answer, just say sorry, I don't know \
        Use three sentences maximum and keep the answer concise.\

        {context}"""

    qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    conversational_rag_chain = RunnableWithMessageHistory(
                                    rag_chain,
                                    get_session_history,
                                    input_messages_key="input",
                                    history_messages_key="chat_history",
                                    output_messages_key="answer",
                                )

    result = conversational_rag_chain.invoke(
        {"input": input},
        config={
            "configurable": {"session_id": "abc123"}
        },  # constructs a key "abc123" in `store`.
    )
    return result['answer']



def get_session_history(session_id: str) -> BaseChatMessageHistory:
    store = {}
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index.html')


@app.route('/read_info', methods=["GET", "POST"])
def user_input():
    if request.method == 'POST':
        user_input = request.form['question']
        if validate_url(user_input):
            extract_text(user_input)
            os.system("python store_index.py")

    return jsonify({"response": str(user_input)})


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg

    if input == "exit":
        clear_previous_data()
        return "Thanks for contacting, please paste the new web URL if you want to continue"
    else:
        result = qa(input)
        return str(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)