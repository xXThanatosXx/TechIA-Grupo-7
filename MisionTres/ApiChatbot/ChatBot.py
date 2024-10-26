import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from deep_translator import GoogleTranslator
from docx import Document  # Importar la biblioteca python-docx

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# Función para leer el contenido de un archivo .docx
def read_docx(file_path):
    doc = Document(file_path)
    full_text = []
    for paragraph in doc.paragraphs:
        full_text.append(paragraph.text)
    return "\n".join(full_text)

# Función para leer múltiples archivos .docx
def read_multiple_docx(file_paths):
    combined_text = []
    for file_path in file_paths:
        text = read_docx(file_path)
        combined_text.append(text)
    return "\n".join(combined_text)

# Lista de archivos .docx a leer
file_paths = [
    "D:/Shadow/GitHub/TechIA-Grupo-7/MisionTres/ApiChatbot/entrenamiento.docx",
    "D:/Shadow/GitHub/TechIA-Grupo-7/MisionTres/ApiChatbot/entrenamiento2.docx",
    #"D:/Shadow/GitHub/TechIA-Grupo-7/MisionTres/NLTK/entrenamiento3.docx"
]

# Leer el contenido de los múltiples archivos .docx
texto = read_multiple_docx(file_paths)

# Preprocesamiento: Tokenización de oraciones y palabras, lematización y eliminación de stopwords
sentences = sent_tokenize(texto)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

preprocessed_sentences = [preprocess_text(sentence) for sentence in sentences]

# Crear un vectorizador TF-IDF para calcular las similitudes entre las preguntas y las oraciones preprocesadas
vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # Usar unigramas y bigramas
vectorized_sentences = vectorizer.fit_transform(preprocessed_sentences)

# Función para traducir la pregunta al inglés utilizando deep-translator
def translate_question(question):
    try:
        translation = GoogleTranslator(source='es', target='en').translate(question)
        return translation
    except Exception as e:
        print(f"Error al traducir: {e}")
        return None

# Función para buscar las oraciones más relevantes basadas en una pregunta traducida
def find_most_relevant_sentences(question, top_n=3, similarity_threshold=0.2):
    preprocessed_question = preprocess_text(question)
    vectorized_question = vectorizer.transform([preprocessed_question])
    similarities = cosine_similarity(vectorized_sentences, vectorized_question).flatten()

    # Obtener los índices de las oraciones más similares, ordenados por similitud
    relevant_indices = similarities.argsort()[-top_n:][::-1]
    relevant_sentences = [sentences[i] for i in relevant_indices if similarities[i] >= similarity_threshold]

    # Si no se encuentra ninguna oración relevante, usar una respuesta por defecto
    if not relevant_sentences:
        return ["Lo siento, no tengo una respuesta para eso."]

    return relevant_sentences

# Loop para hacer preguntas desde la línea de comandos
while True:
    question = input("Hola soy MiniGpt en qué puedo ayudarte?")
    if question.lower() == "salir":
        break
    translated_question = translate_question(question)
    if translated_question is None:
        print("No se pudo traducir la pregunta. Por favor, inténtalo de nuevo.")
        continue

    relevant_sentences = find_most_relevant_sentences(translated_question)

    # Combinar las respuestas más relevantes
    combined_response = " ".join(relevant_sentences)

    # Traducir la respuesta combinada al español usando deep-translator
    try:
        translated_text = GoogleTranslator(source='en', target='es').translate(combined_response)
        print("Respuesta:", translated_text)
    except Exception as e:
        print(f"Error al traducir la respuesta: {e}")
