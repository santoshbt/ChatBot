from langchain.vectorstores import Chroma
from src.helper import load_embedding
from dotenv import load_dotenv
import os
from src.helper import extract_text, validate_url, clear_previous_data
from flask import Flask, render_template, jsonify, request
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate


app = Flask(__name__)

load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embeddings = load_embedding()
persist_directory = "db"
# Now we can load the persisted database from disk, and use it as normal.
vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embeddings)

custom_prompt_template = """
You are a helpful assistant who answer politely to the customer queries.
Focus only on the information related to insurance domain.
Always answer in a step by step manner.
if any context query is asked, respond as Sorry, I am not aware of it.
{context}

Question: {question}
"""

custom_prompt = PromptTemplate(
    template=custom_prompt_template,
    input_variables=["context", "question"],
)

llm = ChatOpenAI()
memory = ConversationSummaryMemory(llm=llm, memory_key = "chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(llm,
                                           retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={"k":3}),
                                           combine_docs_chain_kwargs={"prompt": custom_prompt},
                                           memory=memory)


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
        return str(result["answer"])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)