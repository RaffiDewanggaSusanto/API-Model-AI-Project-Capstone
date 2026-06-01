import os, tempfile
from pathlib import Path
from flask import Flask, jsonify, request
from inference import predict_image, load_model
import google.generativeai as genai

app = Flask(__name__)

# Gemini untuk info edukatif
genai.configure(api_key=os.environ.get('GEMINI_API_KEY', ''))
gemini_model = genai.GenerativeModel('gemini-2.0-flash')

WASTE_CONTEXT = {
    'Kaca'   : 'sampah kaca seperti botol, gelas, atau cermin',
    'Kardus' : 'sampah kardus atau karton bekas kemasan',
    'Kertas' : 'sampah kertas seperti koran, buku, atau kertas bekas',
    'Logam'  : 'sampah logam seperti kaleng aluminium atau besi',
    'Plastik': 'sampah plastik seperti botol, kantong, atau wadah plastik',
    'Residu' : 'sampah residu yang sulit didaur ulang',
}

def get_education(waste_class: str, confidence: float) -> str:
    try:
        desc   = WASTE_CONTEXT.get(waste_class, waste_class)
        prompt = f"""Kamu adalah asisten edukasi pengelolaan sampah untuk aplikasi di Indonesia.
Model AI mendeteksi '{desc}' dengan keyakinan {confidence*100:.1f}%.
Berikan panduan singkat Bahasa Indonesia: cara daur ulang (2-3 poin),
dampak lingkungan (1 kalimat), fakta menarik (1 kalimat). Max 130 kata."""
        return gemini_model.generate_content(prompt).text
    except Exception:
        return WASTE_CONTEXT.get(waste_class, '')

@app.get('/health')
def health():
    return jsonify({'status': 'ok', 'model': 'waste_classifier_final'})

@app.get('/classes')
def get_classes():
    from config import CLASS_NAMES
    return jsonify({'classes': CLASS_NAMES})

@app.post('/predict')
def predict():
    if 'image' not in request.files:
        return jsonify({'error': "Field harus bernama 'image'"}), 400

    f      = request.files['image']
    suffix = Path(f.filename or 'img.jpg').suffix or '.jpg'

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        f.save(tmp.name)
        tmp_path = tmp.name

    try:
        predictions    = predict_image(open(tmp_path, 'rb').read(), top_k=3)
        top_prediction = predictions[0]
        education      = get_education(
            top_prediction['label'],
            top_prediction['confidence']
        )
        return jsonify({
            'prediction' : top_prediction,
            'top_3'      : predictions,
            'education'  : education,
            'disclaimer' : 'Hasil klasifikasi adalah rekomendasi AI, bukan keputusan mutlak.',
        })
    finally:
        os.remove(tmp_path)

# Load model saat startup supaya request pertama tidak lambat
with app.app_context():
    load_model()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)