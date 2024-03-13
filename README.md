# ESchatbot

This is a CLI chatbot that uses langchain RAG model to answer questions. It also uses a Chroma DB to store and retrieve data.


Install dependencies.

```python
pip install -r requirements.txt
```

Create the Chroma DB.

```python
python db.py
```

Query the Chroma DB.

```python
python chatbot.py "How does Alice meet the Mad Hatter?"
```

You'll also need to set up an OpenAI account (and set the OpenAI key in your environment variable) for this to work.