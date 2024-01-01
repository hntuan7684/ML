# Book Recommendation Engine using KNN

import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Sample data (you can replace this with your dataset)
data = {
    'Book': ['Book1', 'Book2', 'Book3', 'Book4', 'Book5'],
    'User1': [5, 4, 0, 3, 2],
    'User2': [0, 2, 4, 5, 1],
    'User3': [2, 0, 5, 4, 3],
}

df = pd.DataFrame(data)

# Drop the 'Book' column for training
X = df.drop('Book', axis=1)

# Initialize the KNN model
knn_model = NearestNeighbors(n_neighbors=2, metric='cosine')
knn_model.fit(X)

# Function to get book recommendations for a given user
def get_book_recommendations(user_ratings, model, books):
    _, indices = model.kneighbors([user_ratings], 3)  # Adjust the number of neighbors as needed
    recommended_books = [books.iloc[i]['Book'] for i in indices.flatten()]
    return recommended_books

# Example: Get book recommendations for User1
user1_ratings = [5, 4, 0, 3, 2]
recommended_books_user1 = get_book_recommendations(user1_ratings, knn_model, df)
print(f"Recommended books for User1: {recommended_books_user1}")
