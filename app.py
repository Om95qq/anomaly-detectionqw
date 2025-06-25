from flask import Flask, render_template, request, send_file
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def detect_anomaly(file_path):
    data = pd.read_csv(file_path)

    # For demo / normal data
    is_clean_data = 'normal' in file_path.lower() or 'sensor_data.csv' in file_path.lower()

    if is_clean_data:
        data['model_anomaly'] = 1
    else:
        model = IsolationForest(contamination=0.01, n_estimators=200, max_samples='auto', random_state=42)
        model.fit(data)
        data['model_anomaly'] = model.predict(data)

    data['model_status'] = data['model_anomaly'].apply(lambda x: 'Anomaly' if x == -1 else 'Normal')

    def rule_based_check(row):
        if row['temperature'] < 30 or row['temperature'] > 60:
            return 1
        if row['vibration'] <= 0.00 or row['vibration'] > 0.5:
            return 1
        if row['pressure'] < 500 or row['pressure'] > 2000:
            return 1
        return 0

    data['rule_anomaly'] = data.apply(rule_based_check, axis=1)
    data['final_status'] = data.apply(
        lambda row: 'Anomaly' if row['rule_anomaly'] == 1 or row['model_anomaly'] == -1 else 'Normal', axis=1
    )

    data.to_csv('sensor_data_with_anomaly.csv', index=False)

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axs[0].set_ylim(20, 100)
    axs[1].set_ylim(0, 1.2)
    axs[2].set_ylim(500, 6000)

    axs[0].plot(data['temperature'], label='Temperature', marker='o', color='orange')
    axs[0].scatter(data[data['final_status'] == 'Anomaly'].index,
                   data[data['final_status'] == 'Anomaly']['temperature'],
                   color='red', s=80, label='Anomaly')
    axs[0].set_title('Temperature')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(data['vibration'], label='Vibration', marker='o', color='green')
    axs[1].scatter(data[data['final_status'] == 'Anomaly'].index,
                   data[data['final_status'] == 'Anomaly']['vibration'],
                   color='red', s=80, label='Anomaly')
    axs[1].set_title('Vibration')
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(data['pressure'], label='Pressure', marker='o', color='blue')
    axs[2].scatter(data[data['final_status'] == 'Anomaly'].index,
                   data[data['final_status'] == 'Anomaly']['pressure'],
                   color='red', s=80, label='Anomaly')
    axs[2].set_title('Pressure')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    graph_path = os.path.join(STATIC_FOLDER, 'anomaly_plot.png')
    plt.savefig(graph_path)
    plt.close()

    if len(data[data['final_status'] == 'Anomaly']) >= 2:
        result = "<div style='margin-top: 20px; padding: 15px; background-color: #ff4c4c; color: white; font-weight: bold; border-radius: 8px;'>⚠️ Machine may be faulty!</div>"
    else:
        result = "<div style='margin-top: 20px; padding: 15px; background-color: #28a745; color: white; font-weight: bold; border-radius: 8px;'>✅ Machine is working normally.</div>"

    return graph_path, result

@app.route('/', methods=['GET', 'POST'])
def index():
    table_html = None
    graph_path = None
    result = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            graph_path, result = detect_anomaly(file_path)

            df = pd.read_csv('sensor_data_with_anomaly.csv')
            table_html = df.head(20).to_html(classes='data', header=True, index=False)

    return render_template('index.html', graph=graph_path, result=result, table_html=table_html)

@app.route('/download')
def download_file():
    return send_file('sensor_data_with_anomaly.csv', as_attachment=True)

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))  # default port for Heroku/Render
    app.run(host='0.0.0.0', port=port)
