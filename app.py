from flask import Flask, request, render_template, redirect, url_for
import os
from main import DocumentAssistant

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './assets'

assistant = DocumentAssistant()
assistant.setup()

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "pdf" not in request.files:
        return redirect(url_for("index"))
    file = request.files["pdf"]
    if file.filename == "":
        return redirect(url_for("index"))
    if file and file.filename.lower().endswith(".pdf"):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        assistant.vector_store_manager.upload_pdf_files(assistant.vector_store_id)
    return redirect(url_for("index"))

@app.route("/ask", methods=["POST"])
def ask():
    question = request.form.get("question")
    if not question:
        return render_template("index.html", answer="Please enter a question.")
    
    assistant.ask_question(question)
    messages = assistant.assistant_manager.get_messages(assistant.thread_id, limit=1)
    if messages and messages[0].role == "assistant":
        answer = messages[0].content[0].text.value
        return render_template("index.html", answer=answer)
    else:
        return render_template("index.html", answer="No response from assistant.")

if __name__ == "__main__":
    app.run(debug=True)
