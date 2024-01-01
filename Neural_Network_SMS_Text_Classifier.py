import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample data (you can replace this with your dataset)
data = {
    'SMS': ['Hey, how are you?', 'WINNER! You have won a prize.', 'Free discount - Limited time offer', 'Meeting at 3 pm', 'URGENT: Your account needs attention.'],
    'Label': ['ham', 'spam', 'spam', 'ham', 'spam'],
}

df = pd.DataFrame(data)

# Preprocess the data
max_words = 1000
max_sequence_length = 50

tokenizer = Tokenizer(num_words=max_words, split=' ')
tokenizer.fit_on_texts(df['SMS'].values)
X = tokenizer.texts_to_sequences(df['SMS'].values)
X = pad_sequences(X, maxlen=max_sequence_length)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['Label'].values)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the neural network model
model = Sequential()
model.add(Embedding(max_words, 32, input_length=max_sequence_length))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')

# Example: Make predictions for new SMS messages
new_messages = ['Hello, how are you doing today?', 'You have won a free gift. Click here to claim.']
sequences = tokenizer.texts_to_sequences(new_messages)
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
predictions = model.predict(padded_sequences)

for i, message in enumerate(new_messages):
    print(f'Message: {message}')
    print(f'Predicted Label: {"spam" if predictions[i] > 0.5 else "ham"}')
    print()
