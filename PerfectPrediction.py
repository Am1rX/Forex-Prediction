import os
import subprocess
try:
    os.remove("missing_libraries.txt")
    os.remove("nozari.bat")
except:
    pass
libraries = ['os', 'pandas', 'numpy', 'joblib', 'sklearn.ensemble', 'sklearn.model_selection', 
             'matplotlib.pyplot', 'plotly.graph_objs', 'plotly.offline', 'datetime', 'platform']
missing_libraries = []
for lib in libraries:
    try:
        __import__(lib)
    except ImportError:
        missing_libraries.append(lib)
if len(missing_libraries) != 0 :
    f = open('missing_libraries.txt', 'w')
    f.write('The following libraries are missing:\n\n')
    for lib in missing_libraries:
        f.write(lib + '\n')
    f.close()        
if 'missing_libraries.txt' in os.listdir('.'):
    installing = "pip install pandas joblib numpy matplotlib plotly scikit-learn\npython PerfectPrediction.py"
    bat = open('nozari.bat', 'w')
    bat.write(installing)
    bat.close()
    subprocess.run(['nozari.bat'])
    print("Wait for installing requared libraries . . .\nAfter that start the script agian.")
else:
    import os
    import pandas as pd
    import numpy as np
    import joblib
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import plotly.graph_objs as go
    import plotly.offline as pyo
    import datetime
    now = datetime.datetime.now()
    stime = now.strftime("%Y-%m-%d %H:%M:%S")
    if not os.path.isfile("model.pkl"):
        f = open("model.pkl", 'wb')
        f.close()
    def load_data(file_path):
        try:
            data = pd.read_csv(file_path, header=None)
        except FileNotFoundError:
            print(f"Error: could not find data file '{file_path}'")
            exit()
        data.columns = ["Timestamp", "Open", "High", "Low", "Close"]
        data["Timestamp"] = pd.to_datetime(data["Timestamp"], format="%m/%d/%Y %I:%M:%S %p")
        data.set_index("Timestamp", inplace=True)
        return data
    def generate_input_features_target_variable(data, window_size):
        X = np.zeros((len(data) - window_size, window_size, 4))
        y = np.zeros((len(data) - window_size,))
        for i in range(window_size, len(data)):
            X[i-window_size] = data.iloc[i-window_size:i].values
            y[i-window_size] = data.iloc[i]["Close"]
        return X.reshape(-1, window_size*4), y
    def train_model(X, y):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model
    def save_model(model, model_file):
        try:
            joblib.dump(model, model_file)
        except IOError:
            print(f"Error: could not write to model file '{model_file}'")
            exit()
    def make_predictions(model, last_candles, num_predictions):
        predictions = []
        for i in range(num_predictions):
            prediction = model.predict(last_candles.reshape(1, -1))
            predictions.append(prediction[0])
            last_candles[:-1] = last_candles[1:]
            last_candles[-1] = np.append(last_candles[-1, 1:], prediction)
        return predictions
    def plot_predictions(predictions, data):
        x = np.arange(len(data), len(data) + len(predictions))
        predicted_close = predictions
        predicted_open = [data.iloc[-1]["Close"]] + predicted_close[:-1]
        predicted_high = [max(o, c) for o, c in zip(predicted_open, predicted_close)]
        predicted_low = [min(o, c) for o, c in zip(predicted_open, predicted_close)]
        trace = go.Candlestick(x=x,
                            open=predicted_open,
                            high=predicted_high,
                            low=predicted_low,
                            close=predicted_close)
        data = [trace]
        layout = go.Layout(title="Nozari Prediction From BigData")
        fig = go.Figure(data=data, layout=layout)
        pyo.plot(fig, filename='nozari.html', auto_open=True)
    if __name__ == "__main__":
        data_file = input("[User] Enter Your CSV file name with format [exp: test.csv] >> ")
        model_file = "model.pkl"
        window_size = 10
        num_predictions = input("[User] Enter Your Prediction candel numbers [exp: 10] >> ")
        print("[+] Prediction Smart-Money has been started.\n[+] Just wait 3 min . . .")
        num_predictions = int(num_predictions)
        data = load_data(data_file)
        X, y = generate_input_features_target_variable(data, window_size)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = train_model(X_train, y_train)
        save_model(model, model_file)
        loaded_model = joblib.load(model_file)
        last_candles = data.tail(window_size).values
        predictions = make_predictions(loaded_model, last_candles, num_predictions)
        plot_predictions(predictions, data)
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        print(f"[+] Train Score: {train_score:.4f}")
        print(f"[+] Test Score: {test_score:.4f}")
        print("\n[+] Prediction html file has been created on the same folder.")
# t.me/str4n5er