import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
model_ckpt ="distilbert/distilbert-base-uncased"
model_name =f"{model_ckpt}-finetuned-emotions-shixm"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
def predict(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return predictions


from flask import Flask, request, jsonify
import torch

app = Flask(__name__)

def inferExec(inputs):
    map_label = {0: '悲伤', 1: '喜悦', 2: '爱', 3: '愤怒', 4: '恐惧', 5: '惊喜'}
    resu = predict(inputs, model, tokenizer)
    max = torch.argmax(resu)
    print(f"resu:{resu},max:{max}")
    return map_label[max.item()]

@app.route('/predict', methods=['POST'])
def predict_text():
    data = request.get_json()
    text = data.get('text')
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # predictions = predict(text, model, tokenizer)
    ret = inferExec(text)
    return jsonify(ret)


if __name__ == '__main__':
    print(inferExec("fuck！@@@"))
    print(inferExec("iamhappy！@@@"))
    app.run(debug=True)

