import spacy
import cupy
import torch

print('torch cuda available: ', torch.cuda.is_available())
print('spacy require gpu: ', spacy.require_gpu())
print('cupy: ', cupy.zeros(10))

nlp = spacy.load("en_core_web_trf")
doc = nlp("Apple shares rose on the news. Apple pie is delicious.")
print([(ent.text, ent.label_) for ent in doc.ents])