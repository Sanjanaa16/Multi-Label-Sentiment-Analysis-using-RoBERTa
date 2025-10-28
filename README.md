 GoEmotions Multilabel Sentiment Analysis using RoBERTa
This project fine-tunes RoBERTa-base on the GoEmotions dataset — a multilabel emotion classification dataset originally curated by Google AI, containing 58k Reddit comments labeled with 27 emotion classes plus neutral.
The goal is to predict multiple emotions per text input (since one sentence can express more than one emotion).

Project Structure
goemotions-multilabel/ ├── README.md ├── requirements.txt ├── sentiment_analysis_project.ipynb # Main Colab notebook └── sentiment_model_roberta/ └── checkpoint-10854/ # Final trained model checkpoint

yaml Copy code

#Model Details

Parameter	Value
Base Model	roberta-base
Task	Multilabel emotion classification
Dataset	GoEmotions
No. of Labels	28 (27 emotions + neutral)
Loss Function	BCEWithLogitsLoss
Optimizer	AdamW
Learning Rate	2e-5
Batch Size	16
Max Sequence Length	128
Epochs	2
Framework	Hugging Face Transformers + PyTorch
Device	Google Colab GPU
Training & Evaluation Results
Metric	Value
Training Loss	0.108
Eval Loss	0.084
Accuracy (Top-1)	0.457
F1 Micro	0.577
F1 Macro	0.400
TensorBoard logs are saved under:
/runs/Oct28_13-29-26_670d1fb31215/events.out.tfevents...

Example Predictions
Text: "I've never been this sad in my life!" Predicted: sadness 😢 Actual: sadness 😢

Text: "Good. We don't want more thrash liberal offspring in this world." Predicted: anger 😡 Actual: anger 😡

Text: "Thank you for your vote of confidence!" Predicted: gratitude 🙏, joy 😊 Actual: gratitude 🙏, joy 😊

yaml Copy code

How to Run the Project
1️ Clone the Repository
git clone https://github.com/USERNAME/goemotions-multilabel.git
cd goemotions-multilabel
2️ Install Dependencies
bash
Copy code
pip install -r requirements.txt
3️ Open the Notebook
Run the notebook in Google Colab or Jupyter:

bash
Copy code
jupyter notebook sentiment_analysis_project.ipynb
 Using the Trained Model for Inference
python
Copy code
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_path = "./sentiment_model_roberta/checkpoint-10854"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

text = "I'm feeling amazing today!"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
predictions = torch.sigmoid(outputs.logits)

# Threshold emotions with score > 0.5
predicted_labels = (predictions > 0.5).nonzero(as_tuple=True)[1].tolist()
print("Predicted emotion indices:", predicted_labels)
 Requirements
Make sure you have the following Python packages installed (see requirements.txt for exact versions):

nginx
Copy code
transformers
datasets
torch
numpy
pandas
scikit-learn
tqdm
matplotlib
You can install them all at once:

bash
Copy code
pip install -r requirements.txt
Notes
This is a multilabel classification problem (not single-label).
Example: “I’m nervous but also excited” → [anxiety, excitement]

Labels include: admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral.

Model weights are saved under:
sentiment_model_roberta/checkpoint-10854/

 Author
Sanjana Upadhyay
Fine-tuned and evaluated on Google Colab — October 2025
For academic and research use.

 License
MIT License © 2025 Sanjana Upadhyay
Feel free to use, modify, and distribute with attribution.

 If you find this project helpful, consider giving it a star on GitHub!

yaml
Copy code

---

Would you like me to generate the **`requirements.txt`** next (with exact versions used in Colab so anyone cloning can run it smoothly)?  
I can make it precise to your training setup (Transformers 4.x, PyTorch 2.x, etc.).
