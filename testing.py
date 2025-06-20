from pymongo import MongoClient

uri = "mongodb+srv://sarthaksc:jBNJg0nbqW14qvrr@cluster0.v041joh.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri)
db = client.sample_analytics
print(db.customers.count_documents({}))