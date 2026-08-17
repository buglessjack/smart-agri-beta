import os
import io
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify

# Render ပေါ်တွင် Version Mismatch Error မဖြစ်စေရန် စစ်ဆေးရေးဆွဲထားပါသည်
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

app = Flask(__name__)

MODEL_PATH = 'my_custom_plant_model.tflite'

# TFLite Interpreter အား စတင်ပတ်မောင်းခြင်း
if os.path.exists(MODEL_PATH):
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("🚀 TFLite AI Model Loaded Successfully!")
else:
    raise FileNotFoundError(f"Error: {MODEL_PATH} ဖိုင်အား ရှာမတွေ့ပါ။")

# အသစ်ပြင်ဆင်ထားသော Disease Class Names များ (အုပ်စု ၃၃ ခု)
class_names = [
    'Alternaria_D', 
    'Anthracnose - Colletotrichum', 
    'Bacterial_spot', 
    'Bacterialblight', 
    'Blast', 
    'Botrytis Leaf Blight', 
    'Brownspot', 
    'Bulb Rot', 
    'Bulb_blight-D', 
    'Caterpillar-P', 
    'Downy mildew', 
    'Early_blight', 
    'Fusarium-D', 
    'Healthy', 
    'Iris yellow virus_augment', 
    'Late_blight', 
    'Leaf curl', 
    'Leaf spot', 
    'Leaf_Mold', 
    'Mosaic_virus', 
    'Purple blotch', 
    'Rust', 
    'Septoria_leaf_spot', 
    'Stemphylium Leaf Blight', 
    'Target_Spot', 
    'Tungro', 
    'Two-spotted_spider_mite', 
    'Virosis-D', 
    'Whitefly', 
    'Xanthomonas Leaf Blight', 
    'Yellow_Leaf_Curl_Virus', 
    'Yellowish', 
    'unknown'
]

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'success': False, 'disease_name': '', 'error': 'No file uploaded.'}), 400
    
    file = request.files['file']

    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB').resize((224, 224))

        # Preprocessing
        x = np.array(img, dtype=np.float32) / 255.0
        x = np.expand_dims(x, axis=0)

        # TFLite Inference တွက်ချက်ခြင်း
        interpreter.set_tensor(input_details[0]['index'], x)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])

        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        predicted_name = class_names[predicted_class_idx]

        # unknown ဖြစ်ပါက
        if predicted_name == 'unknown':
            return jsonify({'success': False, 'disease_name': 'Unknown', 'error': 'ဓာတ်ပုံသည် အပင်ပုံ မဟုတ်ပါဗျာ။'})

        # Confidence Threshold စစ်ဆေးခြင်း
        if confidence < 0.55:
            return jsonify({'success': False, 'disease_name': '', 'error': 'AI မှ သေချာစွာ ခွဲခြားမရပါ။'})

        # 💡 Healthy သို့မဟုတ် healthy ဖြစ်နေပါက disease_name ကို "health" ဟု ပြောင်းလဲပေးခြင်း
        if predicted_name.lower() == 'healthy':
            predicted_name = 'health'

        return jsonify({
            'success': True,
            'disease_name': predicted_name, 
            'confidence': round(confidence * 100, 2),
            'error': ''
        })
    except Exception as e:
        return jsonify({'success': False, 'disease_name': '', 'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
