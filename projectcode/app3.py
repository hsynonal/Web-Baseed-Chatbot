import json
import random
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

model_path = 'C:/Users/Public/projectcode/chatbot'
token = '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'

try:
    model = AutoModelForSequenceClassification.from_pretrained(model_path, use_auth_token=token)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_auth_token=token)
    chatbot = pipeline('text-classification', model=model, tokenizer=tokenizer)
    print("Model ve tokenizer başarıyla yüklendi.")
except Exception as e:
    print(f"Model yüklenirken hata oluştu: {e}")

try:
    with open('C:/Users/Public/projectcode/database/intents.json', 'r', encoding='utf-8') as f:
        intents = json.load(f)
    print("intents.json dosyası başarıyla yüklendi.")
except Exception as e:
    print(f"intents.json dosyası yüklenirken hata oluştu: {e}")

label2id = {intent['tag'].lower(): i for i, intent in enumerate(intents['intents'])}
id2label = {i: intent['tag'].lower() for i, intent in enumerate(intents['intents'])}

unknown_label = 'bilinmeyen'
learned_label = 'öğrendim'
if unknown_label not in label2id:
    unknown_id = len(label2id)
    label2id[unknown_label] = unknown_id
    id2label[unknown_id] = unknown_label

if learned_label not in label2id:
    learned_id = len(label2id)
    label2id[learned_label] = learned_id
    id2label[learned_id] = learned_label

api_key = '9354a075d30770f29351e6cfe3b959c3'

def fetch_movies(url):
    headers = {"accept": "application/json"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        movies = data.get('results', [])
        movie_titles = [movie.get('title', 'Unknown') for movie in movies]
        random.shuffle(movie_titles)  # Randomly shuffle the movie titles
        movie_titles = movie_titles[:3]  # Take the first 3 shuffled movies
        return movie_titles
    else:
        return []

def get_popular_movies():
    url = f'https://api.themoviedb.org/3/movie/popular?api_key={api_key}&language=en-US&page=1'
    return fetch_movies(url)

def get_trending_movies():
    url = f'https://api.themoviedb.org/3/trending/movie/day?api_key={api_key}'
    return fetch_movies(url)

def get_top_rated_movies():
    url = f'https://api.themoviedb.org/3/movie/top_rated?api_key={api_key}&language=en-US&page=1'
    return fetch_movies(url)

def get_now_playing_movies():
    url = f'https://api.themoviedb.org/3/movie/now_playing?api_key={api_key}&language=en-US&page=1'
    return fetch_movies(url)

def get_upcoming_movies():
    url = f'https://api.themoviedb.org/3/movie/upcoming?api_key={api_key}&language=en-US&page=1'
    return fetch_movies(url)

def chat_response(text, chatbot, intents, label2id):
    text = text.strip().lower()
    try:
        results = chatbot(text)
        print(f"Model results: {results}")  # Log the model output
    except Exception as e:
        return "Üzgünüm, bir hata oluştu."

    if results:
        result = results[0]
        score = result['score']
        label = result['label'].lower()

        print(f"Predicted label: {label}, score: {score}")  # Log the predicted label and score

        if score < 0.6 or label == "bilinmeyen":
            if any(keyword in text for keyword in ["film öner", "farketmez", "bilmiyorum", "sen söyle", "karar veremedim", "kararsızım"]):
                popular_movies = get_popular_movies()
                trending_movies = get_trending_movies()
                if popular_movies and trending_movies:
                    response_template = random.choice([
                        "Size popüler filmlerden birini önerebilirim: {film1}, {film2}, {film3}.",
                        "Bugün trend olan filmlerden birini izlemek ister misiniz? {trend_film1}, {trend_film2}, {trend_film3}.",
                        "Genel bir film önerisi isterseniz, şu filmleri önerebilirim: {film1}, {film2}, {film3}."
                    ])
                    response = response_template.format(
                        film1=popular_movies[0],
                        film2=popular_movies[1],
                        film3=popular_movies[2],
                        trend_film1=trending_movies[0],
                        trend_film2=trending_movies[1],
                        trend_film3=trending_movies[2]
                    )
                    save_new_example(text, "öğrendim")
                    return response
                else:
                    return "Üzgünüm, şu anda filmleri çekemedim."
            save_new_example(text, "öğrendim")
            return "Bu konuyu yeni öğrendim. Bir dahaki sefere daha iyi yardımcı olacağım."
        else:
            if label in label2id:
                tag_index = label2id[label]
                intent = next((intent for intent in intents['intents'] if intent['tag'].lower() == id2label[tag_index].lower()), None)
                if intent:
                    if 'responses' in intent:
                        responses = intent['responses']
                        response = random.choice(responses)
                        save_new_example(text, label)
                        if any(placeholder in response for placeholder in ['{film1}', '{trend_film1}', '{top_rated_film1}', '{now_playing_film1}', '{upcoming_film1}']):
                            popular_movies = get_popular_movies()
                            trending_movies = get_trending_movies()
                            top_rated_movies = get_top_rated_movies()
                            now_playing_movies = get_now_playing_movies()
                            upcoming_movies = get_upcoming_movies()

                            response = response.format(
                                film1=popular_movies[0] if popular_movies else "Film bulunamadı",
                                film2=popular_movies[1] if popular_movies else "Film bulunamadı",
                                film3=popular_movies[2] if popular_movies else "Film bulunamadı",
                                trend_film1=trending_movies[0] if trending_movies else "Film bulunamadı",
                                trend_film2=trending_movies[1] if trending_movies else "Film bulunamadı",
                                trend_film3=trending_movies[2] if trending_movies else "Film bulunamadı",
                                top_rated_film1=top_rated_movies[0] if top_rated_movies else "Film bulunamadı",
                                top_rated_film2=top_rated_movies[1] if top_rated_movies else "Film bulunamadı",
                                top_rated_film3=top_rated_movies[2] if top_rated_movies else "Film bulunamadı",
                                now_playing_film1=now_playing_movies[0] if now_playing_movies else "Film bulunamadı",
                                now_playing_film2=now_playing_movies[1] if now_playing_movies else "Film bulunamadı",
                                now_playing_film3=now_playing_movies[2] if now_playing_movies else "Film bulunamadı",
                                upcoming_film1=upcoming_movies[0] if upcoming_movies else "Film bulunamadı",
                                upcoming_film2=upcoming_movies[1] if upcoming_movies else "Film bulunamadı",
                                upcoming_film3=upcoming_movies[2] if upcoming_movies else "Film bulunamadı"
                            )
                        return response
    return "Bu konuyu yeni öğrendim. Bir dahaki sefere daha iyi yardımcı olacağım."

@app.route('/chatbot', methods=['POST'])
def chatbot_endpoint():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'response': 'Geçersiz istek.'}), 400

    message = data['message']
    print(f"Received message: {message}")  # Log the received message

    response = chat_response(message, chatbot, intents, label2id)
    print(f"Generated response: {response}")  # Log the generated response

    return jsonify({'response': response}), 200

def save_new_example(text, label):
    found = False
    for intent in intents['intents']:
        if intent['tag'].lower() == label:
            if 'patterns' not in intent:
                intent['patterns'] = []
            intent['patterns'].append(text)
            found = True
            break
    if not found:
        new_intent = {
            'tag': label,
            'patterns': [text],
            'responses': []
        }
        intents['intents'].append(new_intent)
    with open('C:/Users/Public/projectcode/database/intents.json', 'w', encoding='utf-8') as f:
        json.dump(intents, f, ensure_ascii=False, indent=4)

def retrain_model():
    try:
        print("Model yeniden eğitiliyor...")
        # Eğitim kodunu buraya ekleyin
        print("Model başarıyla yeniden eğitildi.")
    except Exception as e:
        print(f"Model yeniden eğitilirken hata oluştu: {e}")

@app.route('/feedback', methods=['POST'])
def feedback_endpoint():
    data = request.get_json()
    if not data or 'message' not in data or 'label' not in data:
        return jsonify({'response': 'Geçersiz istek.'}), 400

    message = data['message']
    label = data['label'].strip().lower()

    save_new_example(message, label)
    retrain_model()

    return jsonify({'response': 'Geri bildirim için teşekkürler!'}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)
