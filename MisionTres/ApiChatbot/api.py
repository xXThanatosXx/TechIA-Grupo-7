from flask import Flask, request, jsonify
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from deep_translator import GoogleTranslator
from docx import Document
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt_tab')
# Código similar al que ya tienes, ajustado para Flask
app = Flask(__name__)

# Leer archivos de entrenamiento
def read_docx(file_path):
    doc = Document(file_path)
    full_text = []
    for paragraph in doc.paragraphs:
        full_text.append(paragraph.text)
    return "\n".join(full_text)

# Preprocesamiento de los datos
file_paths = [
    "D:/Shadow/GitHub/TechIA-Grupo-7/MisionTres/ApiChatbot/entrenamiento.docx",
    "D:/Shadow/GitHub/TechIA-Grupo-7/MisionTres/ApiChatbot/entrenamiento2.docx",
]

# Función para leer múltiples archivos .docx
def read_multiple_docx(file_paths):
    combined_text = []
    for file_path in file_paths:
        text = read_docx(file_path)
        combined_text.append(text)
    return "\n".join(combined_text)
texto = read_multiple_docx(file_paths)

sentences = sent_tokenize(texto)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

preprocessed_sentences = [preprocess_text(sentence) for sentence in sentences]
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
vectorized_sentences = vectorizer.fit_transform(preprocessed_sentences)

@app.route('/chat', methods=['POST'])
def chat():
    question = request.json.get('question')
    if not question:
        return jsonify({"error": "No question provided"}), 400

    translated_question = GoogleTranslator(source='es', target='en').translate(question)
    preprocessed_question = preprocess_text(translated_question)
    vectorized_question = vectorizer.transform([preprocessed_question])
    similarities = cosine_similarity(vectorized_sentences, vectorized_question).flatten()

    relevant_indices = similarities.argsort()[-3:][::-1]
    relevant_sentences = [sentences[i] for i in relevant_indices if similarities[i] >= 0.2]

    if not relevant_sentences:
        combined_response = "Lo siento, no tengo una respuesta para eso."
    else:
        combined_response = " ".join(relevant_sentences)

    translated_response = GoogleTranslator(source='en', target='es').translate(combined_response)
    return jsonify({"response": translated_response})

if __name__ == '__main__':
    app.run(debug=True)
