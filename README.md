# 🛳️ Ship & Object Detection App

Detect ships, cargo vessels, islands, and clouds in satellite images or videos using YOLOv8 and Streamlit.  
This app gives a professional and detailed detection summary with object lengths, coordinates, weather status, and risk alerts.

---

## 🚀 Features

- 🔍 Detect from **Image or Video**
- 📍 Show **Coordinates** and **Length**
- 🌤️ Weather Status (Cloudy / Clear)
- ⚠️ Risk Alerts (based on ship & cargo density)
- 📊 Detailed Confidence Summary
- 📥 Download labeled detection output

---

## 🧠 Model Info

- **Model:** YOLOv8
- **Trained on:** 3054 Satellite Images
- **Epochs:** 500
- **Classes:** `ship`, `cargo`, `island`, `cloud`
- **Framework:** [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)

---

## 📦 Installation

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/ship-detection-app.git
cd ship-detection-app
```
2. **Install Dependencies:** *(Recommended to use a virtual environment)*
```bash
pip install -r requirements.txt
```
3. **Run the App:**
```bash
streamlit run app.py
```

---

## 📁 Project Structure
```bash
ship-detection-app/
│
├── app.py                # Streamlit frontend
├── best.pt               # Trained YOLOv8 model weights
├── requirements.txt      # Required Python packages
└── README.md             # Project documentation
```

---

## 📌 Author
**Zahid Al Noor Himanta** 
Project Coordinator, Programming Club USTC
