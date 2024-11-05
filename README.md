# Breast Cancer Classification with VGG16 🧑‍⚕️🎗️

Welcome to the Breast Cancer Classification project! This repository implements a machine learning pipeline to classify breast cancer images into three categories: **Benign**, **Normal**, and **Malignant**. We use the **VGG16** model for training on a Kaggle dataset to predict the class of cancer images.

🔬 **Goal:** Build an automated system that helps doctors in the early diagnosis of breast cancer using deep learning.

---

## 📺 Watch the Demo Video

Check out the demo video on how this model works:

[![Breast Cancer Classification Demo](https://img.youtube.com/vi/zBihyxkr9mY/0.jpg)](https://youtu.be/zBihyxkr9mY)

---

## 📂 Repository Structure

This project is organized as follows:

- **app/**: Contains the Flask web server and HTML files for the frontend.
- **src/**: Holds the machine learning components, configuration files, and pipeline logic.
  - **BREASTCANCERCLASSIFICATION/**: Core module for breast cancer classification.
  - **config/**: Configuration files like `config.yaml`, `secrets.yaml`, `params.yaml`.
- **dvc.yaml**: Configuration for DVC (Data Version Control) workflows.
- **requirements.txt**: List of Python dependencies.
- **main.py**: The main script that runs the entire pipeline (data ingestion, model training, etc.).
- **README.md**: Project documentation.
- **LICENSE**: MIT License.

---

## 🛠️ Requirements

Before running the project, make sure you have the following tools and libraries:

- Python 3.7+
- Git
- Flask
- TensorFlow
- DVC (optional)

To install the required dependencies, follow the steps below.

---

## 🚀 Getting Started

### Step 1: Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/arhamkhan779/BreastCancerClassification.git
cd BreastCancerClassification
```

### Step 2: Create a Virtual Environment

It's recommended to use a virtual environment to manage dependencies:

```bash
python -m venv venv
source venv/bin/activate  # For Windows, use venv\Scripts\activate
```

### Step 3: Install Dependencies

Install all necessary Python libraries using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Step 4: Run the Main Pipeline

To run the data ingestion, preprocessing, and model training pipeline:

```bash
python main.py
```

This will train the **VGG16** model on the breast cancer dataset and save the trained model.

### Step 5: Start the Flask Web Application

After the model is trained, you can start the Flask server:

```bash
python app/template.py
```

Visit `http://127.0.0.1:5000/` in your browser to interact with the application.

---

## 🔧 Workflow Overview

The project follows a specific workflow for training the model and deploying it using Flask:

1. **Update Configurations**: Modify the following configuration files if needed:
   - `config.yaml`: Update model and pipeline parameters.
   - `secrets.yaml`: (Optional) Store sensitive information like API keys.
   - `params.yaml`: Update hyperparameters for training.

2. **Pipeline Steps**:
   - **Data Ingestion**: Collect and preprocess the breast cancer dataset.
   - **Model Training**: Train the VGG16 model on the dataset.
   - **Prediction**: Use the trained model for prediction.

3. **Flask Server**: The Flask app serves as a web interface for users to upload images and get predictions on the class of breast cancer.

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👩‍💻 Contributing

Contributions are welcome! Feel free to fork the repository, submit issues, or open pull requests.

---

## 📑 References

- [Kaggle Breast Cancer Dataset](https://www.kaggle.com/datasets/)
- [VGG16 Architecture](https://arxiv.org/abs/1409.1556)
- [Flask Documentation](https://flask.palletsprojects.com/)

---

### 🚀 Happy Coding! 😄
