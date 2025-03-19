# RAG-OpenAI

OpenAI Assistant that can answer questions about uploaded documents.
It handles PDF uploads to OpenAI's vector stores and provides an interactive interface
for users to ask questions about the documents.

```
# Create virtual environment
python3 -m venv ./venv

# Activate your virtual environment
source venv/bin/activate

# Install the required packages. For example
pip3 install PyPDF2 tqdm openai dotenv



# Rename the file .env-bup to .env 
# Add your OPENAI_API_KEY to the .env file.

# Run the app
python3 main.py
```