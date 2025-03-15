import tensorflow as tf
from keras import layers, Sequential

infeats, fc_feats, units = 768, 1024, 3
model = Sequential([
    layers.InputLayer(input_shape=(infeats,)),
    layers.Dense(units=fc_feats, activation='relu'),
    layers.Dense(units=fc_feats, activation='relu'),
    layers.Dense(units=units)  # No activation here
])

data = tf.random.uniform((100, 768))
labels = tf.one_hot(tf.range(100) % 3, depth=3)

# Use logits for categorical_crossentropy
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=loss_fn, metrics=['accuracy'])
model.fit(data, labels, epochs=1000, verbose=0, callbacks=[
    tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: print(f"Epoch {epoch}: {logs['loss']}"))
])
model.summary()

accuracy = model.evaluate(data, labels, verbose=0)[1]
print(f"Accuracy: {accuracy}")