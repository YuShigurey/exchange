import * as tf from '@tensorflow/tfjs';


let [infeats, fc_feats, units] = [768, 1024, 3]

const model = tf.sequential(
    {
        layers: [
            tf.layers.inputLayer({inputShape: [infeats]}),
            tf.layers.dense({units: fc_feats, activation: 'relu'}),
            tf.layers.dense({units: fc_feats, activation: 'relu'}),
            tf.layers.dense({units: units})
        ]
    }
);

// loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
const loss_fn = tf.losses.softmaxCrossEntropy;
model.compile({optimizer: 'adam', loss: loss_fn, metrics: ['accuracy']});

const data = tf.randomNormal([100, infeats]);
const labels = tf.oneHot(tf.tensor1d(Array.from({length: 100}, () => Math.floor(Math.random() * units)), 'int32'), units);
await model.fit(data, labels, {epochs: 1000, callbacks: {onEpochEnd: (epoch, logs) => console.log('Epoch:', epoch, 'Loss:', logs.loss)}});
console.log('Training complete');
model.summary();
const accuracy = model.evaluate(data, labels)[1].dataSync();
console.log('Accuracy:', accuracy);



