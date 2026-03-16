<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow"/>
  <img src="https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white" alt="Keras"/>
  <img src="https://img.shields.io/badge/Flask-Web%20App-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask"/>
  <img src="https://img.shields.io/badge/LSTM-NLP-blueviolet?style=for-the-badge" alt="LSTM"/>
</p>

<h1 align="center">⚡Synapse AI — Next Word Prediction</h1>

<p align="center">
  <em>An LSTM-based deep learning model that predicts the next word in a sentence, trained on Shakespeare's <strong>Hamlet</strong> and served via a beautiful Flask web application.</em>
</p>

<p align="center">
  <a href="#-features">Features</a> •
  <a href="#-demo">Demo</a> •
  <a href="#-project-structure">Project Structure</a> •
  <a href="#-tech-stack">Tech Stack</a> •
  <a href="#-installation--setup">Installation</a> •
  <a href="#-usage">Usage</a> •
  <a href="#-model-architecture">Model</a> •
  <a href="#-contributing">Contributing</a>
</p>

---

## 📖 About

**Synapse AI** is a learning-focused project that explores how **Recurrent Neural Networks (RNNs)**, specifically **Long Short-Term Memory (LSTM)** networks, can be used for **next-word prediction** in natural language processing.

The model is trained on Shakespeare's *Hamlet* and served through a sleek, modern Flask web application with dark/light theme support and an interactive prediction demo.

> **⚠️ Note:** This is an educational project. Accuracy is limited due to a small training dataset and constrained compute resources. It is not intended for production use.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧠 **LSTM Model** | Deep learning model trained on Shakespeare's Hamlet for next-word prediction |
| 🌐 **Flask Web App** | Clean, production-style web interface with interactive demo |
| 🌗 **Dark / Light Theme** | Toggle between dark and light modes with persistent preference |
| ⚡ **Real-time Prediction** | Type a sentence and get instant next-word suggestions |
| 📓 **Jupyter Notebook** | Full training pipeline available for learning and experimentation |
| 📊 **Performance Stats** | UI dashboard showing inference latency, model parameters, and accuracy info |

---

## 🎬 Demo

1. Navigate to the app in your browser (`http://localhost:5000`)
2. Type a sentence in the text area (e.g., *"To be or not to"*)
3. Click **"Predict Next Word"**
4. The model will predict and display the most likely next word

---

## 📁 Project Structure

```
Next-Word-Prediction/
│
├── app.py                     # Flask application — serves the model & handles predictions
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation (this file)
├── .gitignore                 # Git ignore rules
│
├── data/
│   └── hamlet.txt             # Training dataset — Shakespeare's Hamlet (~168 KB)
│
├── models/
│   ├── next_word_lstm.h5      # Trained LSTM model (~5.4 MB)
│   └── tokenizer.pickel       # Serialized Keras tokenizer (~187 KB)
│
├── notebook/
│   └── next_word_prediction.ipynb  # Jupyter notebook — full training pipeline
│
├── templates/
│   ├── predict.html           # Main prediction page (Synapse AI UI)
│   └── home.html              # Home page template
│
└── static/
    └── css/
        └── style.css          # Custom CSS — dark/light themes, animations, responsive design
```

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.8+ |
| **Deep Learning** | TensorFlow / Keras |
| **Model Architecture** | LSTM (Long Short-Term Memory) |
| **Web Framework** | Flask |
| **Frontend** | HTML5, CSS3 (custom dark/light theme with animations) |
| **Data Processing** | NumPy, Pandas |
| **Serialization** | Pickle (tokenizer), HDF5 (model) |
| **Notebook** | Jupyter Notebook |

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/atulshahi6310/Next-Word-Prediction.git
   cd Next-Word-Prediction
   ```

2. **Create a virtual environment** *(recommended)*
   ```bash
   python -m venv venv
   source venv/bin/activate        # macOS/Linux
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install tensorflow numpy pandas flask
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open in browser**
   ```
   http://localhost:5000
   ```

---

## 💡 Usage

### Web Interface

- Visit `http://localhost:5000` after starting the server
- Enter a partial sentence in the text area
- Click **"Predict Next Word"** to see the LSTM model's prediction
- Use the **theme toggle** (🌙/☀️) in the navbar to switch between dark and light modes

### Programmatic Usage

You can also use the prediction function directly in Python:

```python
from tensorflow.keras.models import load_model
import pickle

# Load model and tokenizer
model = load_model('models/next_word_lstm.h5')
with open('models/tokenizer.pickel', 'rb') as f:
    tokenizer = pickle.load(f)

# Predict
from app import predict_next_word
result = predict_next_word(model, tokenizer, "To be or not to", max_seq_len=20)
print(f"Predicted next word: {result}")
```

---

## 🧠 Model Architecture

| Property | Details |
|---|---|
| **Architecture** | LSTM (Long Short-Term Memory) |
| **Training Data** | Shakespeare's *Hamlet* (~168 KB text) |
| **Parameters** | ~1.3 Million |
| **Max Sequence Length** | 20 tokens |
| **Tokenizer** | Keras `Tokenizer` (word-level) |
| **Output** | Softmax over vocabulary — predicts most probable next word |
| **Inference Latency** | ~50ms |

### How It Works

1. **Text Preprocessing** — Input text is tokenized using a fitted Keras `Tokenizer`
2. **Sequence Padding** — Token sequences are padded to a fixed length of 19 (max_seq_len - 1)
3. **Model Inference** — The padded sequence is fed into the LSTM model
4. **Word Prediction** — The output softmax layer gives probability distribution over the vocabulary; the word with the highest probability is returned

---

## 📓 Training Notebook

The full model training pipeline is available in [`notebook/next_word_prediction.ipynb`](notebook/next_word_prediction.ipynb). It covers:

- 📥 Loading and preprocessing the Hamlet text corpus
- 🔤 Tokenization and sequence generation
- 🏗️ Building the LSTM architecture
- 🏋️ Training the model
- 💾 Saving the model (`.h5`) and tokenizer (`.pickel`)
- 📈 Evaluating model performance

---

## 🗂 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Renders the main prediction page |
| `POST` | `/predict` | Accepts form data with `text` field, returns prediction |
| `GET` | `/predict` | Renders the prediction page (empty state) |

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

---

## 📜 License

This project is open source and available for educational purposes.

---

## 👤 Author

**Atul Shahi**

- GitHub: [@atulshahi6310](https://github.com/atulshahi6310)

---

<p align="center">
  Made with ❤️ for learning Deep Learning & NLP
</p>
