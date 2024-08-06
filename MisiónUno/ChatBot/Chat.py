import nltk
import webbrowser
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from googletrans import Translator

# Leer el contenido del archivo de texto
with open("D:\Shadow\GitHub\TechIA-Grupo-7\MisiónUno\ChatBot\entrenamiento.txt") as file:
    texto = file.read()

# Preprocesamiento: Tokenización de oraciones y palabras, lematización y eliminación de stopwords
sentences = sent_tokenize(texto)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))  # Se utiliza el idioma inglés para las stopwords

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)
def msg():
    webbrowser.open("https://shorturl.at/fBEO8")
preprocessed_sentences = [preprocess_text(sentence) for sentence in sentences]

# Crear un vectorizador TF-IDF para calcular las similitudes entre las preguntas y las oraciones preprocesadas
vectorizer = TfidfVectorizer()
vectorized_sentences = vectorizer.fit_transform(preprocessed_sentences)

# Función para traducir la pregunta al inglés utilizando Google Translate
def translate_question(question):
    translator = Translator()
    translation = translator.translate(question, src='es', dest='en')
    return translation.text

# Función para buscar la oración más relevante basada en una pregunta traducida
def find_most_relevant_sentence(question):
    preprocessed_question = preprocess_text(question)
    vectorized_question = vectorizer.transform([preprocessed_question])
    similarities = cosine_similarity(vectorized_sentences, vectorized_question)
    most_relevant_index = similarities.argmax()
    most_relevant_sentence = sentences[most_relevant_index]
    return most_relevant_sentence

# Loop para hacer preguntas desde la línea de comandos
while True:
    question = input("Hola soy MiniGpt en qué puedo ayudarte?")
    if question.lower() == "salir":
        break
    translated_question = translate_question(question)
    most_relevant_sentence = find_most_relevant_sentence(translated_question)
    #print("Respuesta:", most_relevant_sentence)
    # Crear una instancia del traductor
    translator = Translator(service_urls=['translate.google.com'])
    # Realizar la traducción
    translation = translator.translate(most_relevant_sentence, src='en', dest='es')
    # Obtener el texto traducido
    translated_text = translation.text
    # Imprimir la traducción
    msg()
    print(translated_text)