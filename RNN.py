import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


texts = [
    "Привет, как дела?",
    "Сегодня хорошая погода",
    "Я люблю программировать",
    "Как дела?",
    "Хорошо, спасибо",
    "А у тебя?",
    "Тоже хорошо",
    "Что ты делаешь?",
    "Я работаю над проектом",
    "Интересно",
]


tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
total_words = len(tokenizer.word_index) + 1


input_sequences = []
for line in texts:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)


max_sequence_len = max([len(x) for x in input_sequences])


padded_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')


X = padded_sequences[:, :-1]
y = padded_sequences[:, -1]


model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
model.add(LSTM(100))
model.add(Dense(total_words, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(X, y, epochs=100, verbose=1)


def generate_text(seed_text, next_words=3):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted_probability = model.predict(token_list)[0]
        predicted_index = tf.math.argmax(predicted_probability).numpy()
        predicted_word = tokenizer.index_word[predicted_index]
        seed_text += " " + predicted_word
    return seed_text


seed_text = "Привет, как"
generated_text = generate_text(seed_text)
print(generated_text)