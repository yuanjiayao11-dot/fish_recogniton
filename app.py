import os
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import json
import matplotlib.pyplot as plt
import io
import base64

from nets.Alexnet import AlexNet_v1, AlexNet_v2
from nets.Vgg import vgg
from nets.Googlenet import GoogLeNet
from nets.Resnet import resnet50
from nets.Mobilenetv2 import MobileNetV2
from nets.Mobilenetv3 import mobilenet_v3_small,mobilenet_v3_large
from nets.Shufflenet import shufflenet_v2_x1_0
from nets.EfficientnetV1 import efficientnet_b0
from nets.EfficientnetV2 import efficientnetv2_s

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # 用于flash消息
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/fish_photos/<path:filename>')
def fish_photos(filename):
    return send_from_directory(r'D:\模式识别\Fish_recognition-main\fish_data\fish_photos', filename)


im_height = 224
im_width = 224
model_name = "Googlenet"  # 可根据需要修改或扩展为选择模型
weights_path = "./save_weights/myGooglenet.h5"

# 加载类别字典
json_path = './class_indices.json'
assert os.path.exists(json_path), f"file: '{json_path}' dose not exist."
with open(json_path, "r") as f:
    class_indict = json.load(f)

# 实例化模型
if model_name == "AlexNet":
    model = AlexNet_v1(im_height=im_height, im_width=im_width, num_classes=9)
elif model_name == "VGG":
    model = vgg(model_name="vgg16", im_height=224, im_width=224, num_classes=9)
elif model_name == "Googlenet":
    model = GoogLeNet(class_num=9, aux_logits=False)
elif model_name == "Resnet":
    model = resnet50(num_classes=9, include_top=True)
elif model_name == "MobilenetV3":
    model = mobilenet_v3_large(input_shape=(im_height, im_width, 3),
                               num_classes=9,
                               include_top=True)
elif model_name == "Shufflenet":
    model = shufflenet_v2_x1_0(input_shape=(im_height, im_width, 3),
                               num_classes=9)
else:
    raise ValueError("Unsupported model_name")

assert os.path.exists(weights_path), f"file: '{weights_path}' dose not exist."
model.load_weights(weights_path, by_name=True)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(image_path):
    img = Image.open(image_path)
    img = img.resize((im_width, im_height))
    img_array = np.array(img) / 255.
    img_array = np.expand_dims(img_array, 0)
    result = np.squeeze(model.predict(img_array))
    predict_class = np.argmax(result)
    return predict_class, result

def plot_probabilities(result):
    plt.figure(figsize=(8,4))
    classes = [class_indict[str(i)] for i in range(len(result))]
    probs = result
    bars = plt.bar(classes, probs, color='skyblue')
    plt.ylim([0,1])
    plt.ylabel('Probability')
    plt.title('Prediction Probabilities')
    plt.xticks(rotation=45, ha='right')
    for bar, prob in zip(bars, probs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{prob:.2f}', ha='center', va='bottom')
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return img_base64

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # 检查是否有文件上传
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            predict_class, result = predict_image(filepath)
            prob = result[predict_class]

            # 检查准确率是否低于0.8
            if prob < 0.8:
                return jsonify({
                    'filename': filename,
                    'predicted_label': '该鱼类未入库',
                    'probability': float(prob),
                    'prob_img': None
                })

            predicted_label = class_indict[str(predict_class)]
            prob_img = plot_probabilities(result)


            # 返回JSON格式，方便前端处理
            return jsonify({
                'filename': filename,
                'predicted_label': predicted_label,
                'probability': float(prob),
                'prob_img': prob_img
            })
        else:
            return jsonify({'error': 'Allowed image types are -> png, jpg, jpeg, gif'}), 400
    return render_template('123.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
    app.run(debug=True)
