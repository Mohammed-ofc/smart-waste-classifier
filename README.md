♻️ Smart Waste Classification System (CNN + EfficientNetB0 + Streamlit)

An AI-powered image classifier that detects whether a waste image is Hazardous, Organic, or Recyclable using deep learning and a clean Streamlit interface — ideal for smart city and sustainability applications.

🚀 Features
Upload a waste image (JPG, PNG)
EfficientNetB0-based CNN model trained on waste dataset
Real-time predictions with class-wise probability
Automatically resized and preprocessed input
Visual prediction results using Streamlit

🛠️ Tech Stack
Python, Streamlit
TensorFlow, Keras (EfficientNetB0)
NumPy, PIL (Image Processing)
Matplotlib (for metrics & evaluation)

📂 Folder Structure
smart_waste_classifier/
├── app.py
├── src/
│   ├── model_train.py
│   ├── evaluate_model.py
│   ├── clean_dataset.py
├── dataset/
│   ├── hazardous/
│   ├── organic/
│   └── recyclable/
├── waste_classifier_model.h5
├── .gitignore
└── README.md

📈 Model Performance
✅ Accuracy: 96% on validation set
✅ Model: EfficientNetB0 pretrained on ImageNet
✅ Evaluation: Confusion Matrix & Classification Report

🚀 How to Run
# Clone the repo and navigate to the folder
git clone https://github.com/Mohammed-ofc/smart-waste-classifier.git
cd smart-waste-classifier

# Create a virtual environment and activate it
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the web app
streamlit run app.py

🙋‍♂️ Created By
Mohammed Salman
📧 mohammed.salman.p.2004@gmail.com
🔗 [LinkedIn](https://www.linkedin.com/in/mohammed-salman-p-484a9431b/)
🌐 [GitHub](https://github.com/Mohammed-ofc)


📜 License
This project is licensed under the MIT License.
