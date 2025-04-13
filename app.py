from flask import Flask, request, jsonify
from helpers import embed, context, chat
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

@app.route("/legalchat", methods=["POST"])
def legal_chat():
    data = request.get_json()
    
    query = data.get("query")
    if not query:
        return jsonify({"error": "Query is required"}), 400

    embedded = embed(query)
    legal_context = context(embedded, top_k=5)
    answer = chat(query, legal_context)

    return jsonify({"answer": answer}), 200

if __name__ == "__main__":
    app.run(debug=True)