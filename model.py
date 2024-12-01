import os
import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore

# =======================
# Positional Encoding
# =======================
def positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angle_rads = pos * angle_rates
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return tf.convert_to_tensor(angle_rads, dtype=tf.float32)

# =======================
# Transformer Model
# =======================
class TransformerModel(tf.keras.Model):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, dff, max_seq_len):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pos_encoding = positional_encoding(max_seq_len, d_model)

        self.encoder = [
            tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model) for _ in range(num_layers)
        ]
        self.dense_ffn = [
            tf.keras.Sequential([tf.keras.layers.Dense(dff, activation='relu'), tf.keras.layers.Dense(d_model)])
            for _ in range(num_layers)
        ]
        self.layer_norm = [tf.keras.layers.LayerNormalization(epsilon=1e-6) for _ in range(num_layers)]
        self.final_dense = tf.keras.layers.Dense(vocab_size)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x) + self.pos_encoding[:seq_len]
        for i in range(len(self.encoder)):
            attn_output = self.encoder[i](x, x)
            x = self.layer_norm[i](x + attn_output)
            ffn_output = self.dense_ffn[i](x)
            x = self.layer_norm[i](x + ffn_output)
        return self.final_dense(x)

# =======================
# Save/Load Helpers
# =======================
def save_model_weights_and_config(model, weights_path, config_path):
    model.save_weights(weights_path)
    with open(config_path, 'w') as f:
        json.dump(model.get_config(), f)

def save_tokenizer(tokenizer, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump({
            'vocab': tokenizer.word_index,
            'tokenizer': tokenizer
        }, f)

# =======================
# Main Execution
# =======================
if __name__ == "__main__":
    # Load and tokenize data
    texts = []
    for file in os.listdir("data"):
        if file.endswith(".txt"):
            with open(os.path.join("data", file), "r", encoding="utf-8") as f:
                texts.extend(f.read().strip().splitlines())

    # Use Tokenizer to treat each word as a token
    vocab_size = 800
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<UNK>")
    tokenizer.fit_on_texts(texts)
    
    # Tokenize the text
    tokenized_texts = tokenizer.texts_to_sequences(texts)

    # Prepare inputs and outputs for training
    inputs = [seq[:-1] for seq in tokenized_texts]
    outputs = [seq[1:] for seq in tokenized_texts]
    
    # Find the maximum sequence length
    max_seq_len = max(len(seq) for seq in inputs)
    
    # Pad sequences to the same length
    inputs = pad_sequences(inputs, padding="post", maxlen=max_seq_len)
    outputs = pad_sequences(outputs, padding="post", maxlen=max_seq_len)
    
    # Build the model
    d_model, num_heads, num_layers, dff = 64, 4, 3, 16
    model = TransformerModel(vocab_size, d_model, num_heads, num_layers, dff, max_seq_len)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Train the model
    model.fit(inputs, outputs, epochs=50, batch_size=20)

    # Save model weights and configuration
    save_model_weights_and_config(model, "model.weights.h5", "model_config.json")
    
    # Save tokenizer and vocabulary mappings
    save_tokenizer(tokenizer, "tokenizer.pkl")


    # Save token-to-index and index-to-token mappings
    with open("mappings.pkl", "wb") as f:
        pickle.dump({
            'token_to_idx': tokenizer.word_index,
            'idx_to_token': {idx: word for word, idx in tokenizer.word_index.items()}
        }, f)