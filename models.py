from keras import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout


def lstm_model(inp, x, y, x_val, y_val, x_test, y_test):

    model = Sequential()
    model.add(Embedding(inp, 64, embeddings_regularizer = 'l2'))
    model.add(Dropout(0.4))
    model.add(LSTM(16, recurrent_regularizer='l2'))
    model.add(Dropout(0.4))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    print(model.summary())

    model.fit(x, y, epochs = 30, batch_size=128, verbose =1, validation_data=(x_val, y_val))

    model.evaluate(x_test,y_test)