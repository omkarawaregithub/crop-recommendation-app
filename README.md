# 🌾 Crop Recommendation System

A machine learning-based web application that recommends suitable crops based on soil characteristics and environmental conditions.

## 📋 Features

- Real-time crop recommendations based on input parameters
- Interactive web interface built with Streamlit
- Batch prediction support through CSV upload
- Visual representation of soil nutrients
- Confidence scores for predictions
- Random Forest model with high accuracy

## 🔧 Technical Stack

- Python 3.8+
- Streamlit for web interface
- Scikit-learn for machine learning
- Pandas for data manipulation
- Matplotlib for visualization
- Joblib for model serialization

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/crop-recommendation-app.git
cd crop-recommendation-app
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## 💻 Usage

1. Train the model (if not already trained):
```bash
python train_model.py
```

2. Run the Streamlit app:
```bash
python -m streamlit run app.py
```

3. Access the web interface at `http://localhost:8501`

## 🌐 Live Demo
Check out the live demo at: https://yourusername
The system accepts the following input parameters:
- Nitrogen (N): 0-140
- Phosphorus (P): 0-140
- Potassium (K): 0-205
- Temperature: 0-50°C
- Humidity: 0-100%
- pH: 0-14
- Rainfall: 0-500mm

## 📁 Project Structure

```
crop-recommendation-app/
├── app.py              # Streamlit web application
├── train_model.py      # Model training script
├── requirements.txt    # Project dependencies
├── data/              # Dataset directory
│   └── Crop_recommendation.csv
└── model/             # Trained model directory
    └── crop_model.joblib
```

## 📝 Dataset

The model is trained on the Crop Recommendation Dataset, which includes:
- Soil parameters (N, P, K, pH)
- Environmental conditions (temperature, humidity, rainfall)
- Corresponding suitable crops

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- Your Name - Initial work

## 📚 Acknowledgments

- Dataset source: [Kaggle Crop Recommendation Dataset](https://www.kaggle.com/atharvaingle/crop-recommendation-dataset)
- Thanks to all contributors