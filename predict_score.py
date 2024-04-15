from transformers import BertForSequenceClassification, BertTokenizer
import torch


def predict_rating(job_description, skills, max_len=500):
    # Load the model and tokenizer
    model_path = "resume_rating_model2"
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)

    # Set the model to evaluation mode
    model.eval()

    # Check if CUDA is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Preprocess the text inputs
    encoding = tokenizer.encode_plus(
        job_description + ' [SEP] ' + skills,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    # Move tensors to the appropriate device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Disable gradient calculation
    with torch.no_grad():
        # Forward pass, get logit predictions
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    # Convert logits to ratings (sigmoid activation for regression output)
    predicted_rating = torch.sigmoid(logits).item() * 5  # Assuming the rating scale is 1-5

    return predicted_rating