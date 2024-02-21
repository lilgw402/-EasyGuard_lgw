from transformers import pipeline

generator = pipeline("text-generation", model="facebook/opt-1.3b")
results = generator("我是中国人")
print(results)
