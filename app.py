import os
import io
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
import tensorflow as tf  # ✅ TensorFlow ကိုသုံးပါ

app = Flask(__name__)

# ==========================================
# 🔐 ENVIRONMENT VARIABLES
# ==========================================
MODEL_PATH = os.environ.get('MODEL_PATH', 'my_custom_plant_model.tflite')
CONFIDENCE_THRESHOLD = float(os.environ.get('CONFIDENCE_THRESHOLD', 0.55))
DEBUG_MODE = os.environ.get('DEBUG_MODE', 'False').lower() == 'true'

print("=" * 50)
print("🔐 Environment Variables:")
print(f"   MODEL_PATH: {MODEL_PATH}")
print(f"   CONFIDENCE_THRESHOLD: {CONFIDENCE_THRESHOLD}")
print(f"   DEBUG_MODE: {DEBUG_MODE}")
print("=" * 50)

# ==========================================
# 📦 LOAD TFLITE MODEL (TensorFlow နဲ့)
# ==========================================

if os.path.exists(MODEL_PATH):
    # ✅ ဒီနေရာမှာ tf.lite.Interpreter ကိုသုံးပါ
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("🚀 TFLite AI Model Loaded Successfully with TensorFlow!")
    print(f"   Input shape: {input_details[0]['shape']}")
    print(f"   Output shape: {output_details[0]['shape']}")
else:
    raise FileNotFoundError(f"Error: {MODEL_PATH} ဖိုင်အား ရှာမတွေ့ပါ။")

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

# ==========================================
# 🟢 ကျန်းမာတဲ့ အုပ်စုများ (Healthy Classes)
# ==========================================
healthy_classes = [
    'Healthy',
    'healthy'  
]

# ==========================================
# 🟡 ရောဂါရှိတဲ့ အုပ်စုများ (Disease Classes)
# ==========================================
disease_classes = [
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
    'Yellowish'
]

# ==========================================
# 🔴 မသိသေးတဲ့ အုပ်စု (Unknown)
# ==========================================
unknown_classes = ['unknown']

print(f"✅ Model loaded with {len(class_names)} classes")
print(f"   🟢 Healthy: {len(healthy_classes)} classes")
print(f"   🟡 Diseases: {len(disease_classes)} classes")
print(f"   🔴 Unknown: {len(unknown_classes)} classes")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({
            'success': False, 
            'disease_name': '', 
            'error': 'No file uploaded.'
        }), 400
    
    file = request.files['file']
    
    try:
        # ၁။ ပုံကိုဖတ်ခြင်း
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB').resize((224, 224))
        
        # ၂။ Preprocessing
        x = np.array(img, dtype=np.float32) / 255.0
        x = np.expand_dims(x, axis=0)
        
        # ၃။ TFLite Inference
        interpreter.set_tensor(input_details[0]['index'], x)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])
        
        # ၄။ ရလဒ်ကိုထုတ်ယူခြင်း
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        predicted_name = class_names[predicted_class_idx]
        
        # ==========================================
        # 🟢 စစ်ဆေးချက်များ (Validation)
        # ==========================================
        
        # ၅။ Unknown ဖြစ်နေရင်
        if predicted_name in unknown_classes:
            return jsonify({
                'success': False,
                'disease_name': 'Unknown',
                'confidence': round(confidence * 100, 2),
                'error': 'ဤဓာတ်ပုံသည် အရွက်ပုံမဟုတ်သောကြောင့် ရောဂါမရှာဖွေနိုင်ပါ။ ကျေးဇူးပြု၍ အရွက်ပုံကိုသာ ရိုက်ကူးပါ။'
            })
        
        # ၆။ Confidence နည်းနေရင် (၅၅% အောက်)
        if confidence < 0.55:
            return jsonify({
                'success': False,
                'disease_name': '',
                'confidence': round(confidence * 100, 2),
                'error': f' AI မှ သေချာစွာ ခွဲခြားမရပါ။ ကျေးဇူးပြု၍ ရှင်းလင်းစွာရိုက်ကူးပေးပါ။'
            })
        
        # ၇။ ကျန်းမာတဲ့အပင်ဖြစ်နေရင်
        if predicted_name in healthy_classes:
            return jsonify({
                'success': True,
                'disease_name': 'Healthy',
                'confidence': round(confidence * 100, 2),
                'error': '',
                'message': 'ဤအပင်သည် ကျန်းမာပါသည်။ ရောဂါမတွေ့ရပါ။'
            })
        
        # ၈။ ရောဂါရှိနေရင် (ပုံမှန်အုပ်စု)
        if predicted_name in disease_classes:
            # ရောဂါအမည်ကို လှပအောင်ပြင်ဆင်ခြင်း
            disease_display = predicted_name.replace('_', ' ').replace(' - ', ' - ')
            
            return jsonify({
                'success': True,
                'disease_name': predicted_name,
                'disease_display': disease_display,
                'confidence': round(confidence * 100, 2),
                'error': '',
                'message': f' ဤအပင်တွင် "{disease_display}" ရောဂါ တွေ့ရှိရပါသည်။'
            })
        
        # ၉။ ကျန်တဲ့အုပ်စုတွေအတွက် (Fallback)
        return jsonify({
            'success': True,
            'disease_name': predicted_name,
            'confidence': round(confidence * 100, 2),
            'error': '',
            'message': f'🔍 ခွဲခြားသတ်မှတ်ချက်: {predicted_name}'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'disease_name': '',
            'error': f'⚠️ System Error: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health Check Endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_classes': len(class_names),
        'classes': class_names
    })

@app.route('/', methods=['GET'])
def home():
    """Home Page"""
    return jsonify({
        'name': 'Plant Disease Detection API',
        'version': '1.0.0',
        'model_classes': len(class_names),
        'endpoints': {
            '/predict': 'POST - Upload image for prediction',
            '/health': 'GET - Check API health',
            '/': 'GET - API Info'
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)