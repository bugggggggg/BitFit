from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig


model_name = 'distilbert-base-uncased'
config = AutoConfig.from_pretrained(model_name, num_labels=2, return_dict=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

print(model)