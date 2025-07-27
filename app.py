from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import joblib

app = Flask(__name__)

# Load mô hình và scaler
model = tf.keras.models.load_model('best_model.keras')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Lấy dữ liệu từ form
            age = float(request.form['age'])
            sex = int(request.form['sex'])
            cp = int(request.form['cp'])
            trestbps = float(request.form['trestbps'])
            chol = float(request.form['chol'])
            fbs = int(request.form['fbs'])
            restecg = int(request.form['restecg'])
            thalach = float(request.form['thalach'])
            exang = int(request.form['exang'])
            oldpeak = float(request.form['oldpeak'])
            slope = int(request.form['slope'])
            ca = int(request.form['ca'])
            thal = int(request.form['thal'])

            # Tạo mảng đầu vào
            input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                                    thalach, exang, oldpeak, slope, ca, thal]])

            # Chuẩn hóa dữ liệu
            input_scaled = scaler.transform(input_data)

            # Dự đoán
            prediction = model.predict(input_scaled)[0][0]
            percent = round(float(prediction) * 100, 2)

            # Kết quả trả về
            result_text = f"Người này có tỷ lệ mắc bệnh tim: {percent}%"

        except Exception as e:
            result_text = f"Lỗi xử lý dữ liệu: {str(e)}"

        return render_template('result.html', result=result_text)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=10000)

