from transformers import pipeline

# 使用预训练模型
nlp = pipeline("ner", model="dslim/bert-base-NER")
text = "What is the box office of Interstellar"
print(nlp(text))
