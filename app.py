import os
import json
import uuid
import pdfplumber
import docx
import chromadb
from google import genai
from google.genai import types
from flask import Flask, request, jsonify, render_template, Response, stream_with_context, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.abspath('uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 1. main.json-லிருந்து API Key எடுத்தல்
def load_api_key():
    json_path = 'main.json'
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get("GEMINI_API_KEY") or data.get("api_key")
    return None

API_KEY = load_api_key()
if not API_KEY:
    raise ValueError("பிழை: main.json கோப்பில் API Key கிடைக்கவில்லை!")

# புதிய GenAI Client துவங்குதல்
client = genai.Client(api_key=API_KEY)

# 2. ChromaDB Setup
CHROMA_PATH = "./chroma_db"
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name="multi_doc_chat")

ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.txt'}

# புதிய Embedding செயல்முறை
def get_gemini_embedding(text):
    try:
        response = client.models.embed_content(
            model="text-embedding-004",
            contents=text
        )
        # முதல் embedding vector-ஐ எடுத்தல்
        return response.embeddings[0].values
    except Exception as e:
        print(f"Embedding error: {e}")
        return []

def extract_and_chunk(file_path, filename):
    ext = os.path.splitext(filename)[1].lower()
    chunks = []
    
    try:
        if ext == '.pdf':
            with pdfplumber.open(file_path) as pdf:
                for idx, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    for i in range(0, len(text), 600):
                        snippet = text[i:i+700].strip()
                        if snippet:
                            chunks.append({
                                'text': snippet,
                                'metadata': {'source': filename, 'page': str(idx + 1)}
                            })
                            
        elif ext in ['.docx', '.doc']:
            doc = docx.Document(file_path)
            for idx, para in enumerate(doc.paragraphs):
                text = para.text.strip()
                if text:
                    chunks.append({
                        'text': text,
                        'metadata': {'source': filename, 'page': f"Para {idx + 1}"}
                    })
                    
        elif ext == '.txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                for idx, i in range(0, len(content), 600):
                    snippet = content[i:i+700].strip()
                    if snippet:
                        chunks.append({
                            'text': snippet,
                            'metadata': {'source': filename, 'page': f"Sec {idx + 1}"}
                        })
    except Exception as e:
        print(f"Error parsing {filename}: {e}")
        
    return chunks

def add_to_vector_db(chunks):
    if not chunks:
        return
    documents = [c['text'] for c in chunks]
    metadatas = [c['metadata'] for c in chunks]
    ids = [str(uuid.uuid4()) for _ in chunks]

    embeddings = []
    for doc in documents:
        emb = get_gemini_embedding(doc)
        embeddings.append(emb)

    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/files/<path:filename>')
def serve_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/upload', methods=['POST'])
def upload_docs():
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
        
    files = request.files.getlist('files')
    uploaded_files = []

    for file in files:
        if file.filename:
            safe_name = secure_filename(file.filename)
            ext = os.path.splitext(safe_name)[1].lower()
            if ext in ALLOWED_EXTENSIONS:
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
                file.save(file_path)
                
                chunks = extract_and_chunk(file_path, safe_name)
                add_to_vector_db(chunks)
                uploaded_files.append(safe_name)

    return jsonify({'message': 'Success', 'files': uploaded_files})

@app.route('/delete-file', methods=['POST'])
def delete_file():
    data = request.json
    filename = data.get('filename')
    if not filename:
        return jsonify({'error': 'Filename required'}), 400

    try:
        collection.delete(where={"source": filename})
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Streaming Chat (SDK Best Practice முறை)
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get('query', '')
    language = data.get('language', 'ta-IN')

    lang_text = "Natural, fluent Tamil (தமிழ்)" if language == 'ta-IN' else "Clear English"

    context_snippets = []
    total_docs = collection.count()

    if total_docs > 0:
        query_embed = get_gemini_embedding(query)
        if query_embed:
            results = collection.query(
                query_embeddings=[query_embed],
                n_results=min(4, total_docs)
            )

            if results and 'documents' in results and len(results['documents']) > 0:
                for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
                    context_snippets.append(f"[Source: {meta['source']} | Page: {meta['page']}]\n{doc}")

    final_context = "\n\n".join(context_snippets).strip()

    if final_context:
        system_instruction = f"""You are an intelligent document assistant.
Answer accurately using ONLY the provided context. Use Markdown formatting.
Always cite the source files and pages at the end like:
Sources: [filename, Page X]

Language: Output strictly in {lang_text}.

Context:
{final_context}"""
    else:
        system_instruction = f"""You are a helpful general assistant.
No documents are uploaded yet. Answer normally without citing any sources or page numbers.
Language: Output strictly in {lang_text}."""

    def generate():
        try:
            # 1. Chat session உருவாக்குதல்
            chat_session = client.chats.create(
                model="gemini-3.5-flash-lite",
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction
                )
            )

            # 2. send_message_stream மூலம் ஸ்ட்ரீம் செய்தல் (பரிந்துரைக்கப்பட்ட முறை)
            response = chat_session.send_message_stream(query)

            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            yield f"\n[Error: {str(e)}]"

    return Response(stream_with_context(generate()), mimetype='text/plain')

if __name__ == '__main__':
    app.run(debug=True, port=5000)