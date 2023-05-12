import pickle

with open('DistanceLabelling/trees/save/0.5_5_1000/embeddings.pkl', 'rb') as f:
    data = pickle.load(f)

print(data)
