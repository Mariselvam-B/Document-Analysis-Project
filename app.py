import os
import json
import pdfplumber
import docx
from google import genai
from google.genai import types
from flask import Flask, request, jsonify, render_template, Response, stream_with_context, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.abspath('uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 1. API Key Setup
def load_api_key():
    env_key = os.environ.get("GEMINI_API_KEY")
    if env_key:
        return env_key
    json_path = 'main.json'
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get("GEMINI_API_KEY") or data.get("api_key")
    return None

API_KEY = load_api_key()
if not API_KEY:
    raise ValueError("GEMINI_API_KEY கிடைக்கவில்லை!")

client = genai.Client(api_key=API_KEY)

# பதிவேற்றப்பட்ட ஆவணங்களின் உரையைச் சேமிக்கும் பட்டியல்
DOCUMENT_PAGES = []
ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.txt'}

def extract_pages(file_path, filename):
    ext = os.path.splitext(filename)[1].lower()
    pages_data = []
    try:
        if ext == '.pdf':
            with pdfplumber.open(file_path) as pdf:
                for idx, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    if text.strip():
                        pages_data.append({
                            'source': filename,
                            'page': str(idx + 1),
                            'text': text.strip()
                        })
        elif ext in ['.docx', '.doc']:
            doc = docx.Document(file_path)
            full_text = "\n".join([p.text.strip() for p in doc.paragraphs if p.text.strip()])
            if full_text:
                pages_data.append({
                    'source': filename,
                    'page': 'Full Document',
                    'text': full_text
                })
        elif ext == '.txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().strip()
                if content:
                    pages_data.append({
                        'source': filename,
                        'page': 'Full Document',
                        'text': content
                    })
    except Exception as e:
        print(f"Error extracting {filename}: {e}")
    return pages_data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/files/<path:filename>')
def serve_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/upload', methods=['POST'])
def upload_docs():
    global DOCUMENT_PAGES
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
                
                extracted = extract_pages(file_path, safe_name)
                # பழைய ஆவணத்தின் அதே பெயர் இருந்தால் நீக்கிவிட்டு புதியதைச் சேர்த்தல்
                DOCUMENT_PAGES = [doc for doc in DOCUMENT_PAGES if doc['source'] != safe_name]
                DOCUMENT_PAGES.extend(extracted)
                
                uploaded_files.append(safe_name)

    print(f"Total document sections ready: {len(DOCUMENT_PAGES)}")
    return jsonify({'message': 'Success', 'files': uploaded_files})

@app.route('/delete-file', methods=['POST'])
def delete_file():
    global DOCUMENT_PAGES
    data = request.json
    filename = data.get('filename')
    if not filename:
        return jsonify({'error': 'Filename required'}), 400

    try:
        DOCUMENT_PAGES = [item for item in DOCUMENT_PAGES if item['source'] != filename]
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get('query', '')
    language = data.get('language', 'ta-IN')

    lang_text = "Natural, fluent Tamil (தமிழ்)" if language == 'ta-IN' else "Clear English"

    context_snippets = []
    for item in DOCUMENT_PAGES:
        context_snippets.append(f"[Document: {item['source']} | Page: {item['page']}]\n{item['text']}")

    final_context = "\n\n---\n\n".join(context_snippets).strip()
    print(f"Final Context Length: {len(final_context)} characters")

    if final_context:
        system_instruction = f"""You are an intelligent document assistant.
Analyze and answer accurately using the provided document text below. Use Markdown formatting.
Whenever you provide facts or answers, always cite the source file and page at the end of that section like:
Sources: [filename, Page X]

Language Rule: Output strictly in {lang_text}.

Available Documents:
{final_context}"""
    else:
        system_instruction = f"""You are a helpful general assistant.
No documents are uploaded yet. Answer normally without citing any sources or page numbers.
Language Rule: Output strictly in {lang_text}."""

    def generate():
        models_to_try = ['gemini-3.5-flash-lite']
        for model_name in models_to_try:
            try:
                chat_session = client.chats.create(
                    model=model_name,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction
                    )
                )
                response = chat_session.send_message_stream(query)
                for chunk in response:
                    if chunk.text:
                        yield chunk.text
                return
            except Exception as e:
                if "503" in str(e):
                    continue
                else:
                    yield f"\n[Error: {str(e)}]"
                    return
        yield "\n[Error: சர்வர் பயன்பாட்டு சுமை அதிகமாக உள்ளது. சிறிது நேரம் கழித்து முயற்சிக்கவும்.]"

    return Response(stream_with_context(generate()), mimetype='text/plain')

if __name__ == '__main__':
    app.run(debug=True, port=5000)