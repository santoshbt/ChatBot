from langchain.vectorstores import Chroma
from src.helper import load_embedding
from dotenv import load_dotenv
import os
from src.helper import extract_text, validate_url
from flask import Flask, render_template, jsonify, request
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain


app = Flask(__name__)

load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embeddings = load_embedding()
persist_directory = "db"
# Now we can load the persisted database from disk, and use it as normal.
vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embeddings)


llm = ChatOpenAI()
memory = ConversationSummaryMemory(llm=llm, memory_key = "chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(llm, retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={"k":3}), memory=memory)


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
    print(input)

    if input == "clear":
        with open('insurance_products.txt', 'w') as file:
            file.truncate()
        return "Please enter a new website URL."
    else:
        result = qa(input)
        print(result['answer'])
        return str(result["answer"])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)