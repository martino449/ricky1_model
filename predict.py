import json
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
from model import TransformerModel
import numpy as np

# =======================
# Load Helpers
# =======================
def load_tokenizer(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    tokenizer = data['tokenizer']
    return tokenizer

def load_mappings(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def load_model(weights_path, config_path, vocab_size, d_model, num_heads, num_layers, dff, max_seq_len):
    with open(config_path, 'r') as f:
        config = json.load(f)
    model = TransformerModel(vocab_size, d_model, num_heads, num_layers, dff, max_seq_len)
    model.build(input_shape=(None, max_seq_len))
    model.load_weights(weights_path)
    return model

# =======================
# Penalize Repeated Tokens
# =======================
def penalize_repeated_tokens(predictions, history, penalty=1.2):
    """
    Penalizza i token ripetuti abbassandone la probabilità.
    
    Args:
        predictions: Vettore di logit o probabilità da cui generare il prossimo token.
        history: Sequenza dei token generati finora.
        penalty: Fattore di penalità applicato ai token ripetuti.
    
    Returns:
        probabilities: Vettore di probabilità aggiornato.
    """
    predictions = np.array(predictions)
    for token in history[-1:]:  # Penalizza solo l'ultimo token generato
        predictions[token] /= penalty
    probabilities = np.exp(predictions) / np.sum(np.exp(predictions))
    return probabilities

# =======================
# Predict Next Tokens
# =======================
def predict_next_token(model, tokenized_input, token_to_idx, idx_to_token, max_seq_len, penalty=1.2):
    input_seq_padded = pad_sequences([tokenized_input], maxlen=max_seq_len, padding="post")
    logits = model(input_seq_padded)
    predictions = logits[0, len(tokenized_input) - 1].numpy()
    probabilities = penalize_repeated_tokens(predictions, tokenized_input, penalty=penalty)
    next_token = np.random.choice(len(probabilities), p=probabilities)
    tokenized_input.append(next_token)
    if len(tokenized_input) > max_seq_len:
        tokenized_input = tokenized_input[1:]
    predicted_token = idx_to_token.get(next_token, "<UNK>")
    return predicted_token, tokenized_input

def responser(number, model, tokenized_seed, token_to_idx, idx_to_token, max_seq_len, penalty=1.2):
    response = []
    for _ in range(number):
        predicted_token, tokenized_seed = predict_next_token(
            model, tokenized_seed, token_to_idx, idx_to_token, max_seq_len, penalty=penalty
        )
        response.append(predicted_token)
    return response

# =======================
# Main Execution
# =======================
if __name__ == "__main__":
    tokenizer = load_tokenizer("tokenizer.pkl")
    mappings = load_mappings("mappings.pkl")
    token_to_idx = mappings['token_to_idx']
    idx_to_token = mappings['idx_to_token']

    vocab_size = len(tokenizer.word_index)
    d_model, num_heads, num_layers, dff = 64, 4, 3, 16
    max_seq_len = 50

    model = load_model("model.weights.keras", "model_config.json", vocab_size, d_model, num_heads, num_layers, dff, max_seq_len)

    seed_text = "Le"
    tokenized_seed = tokenizer.texts_to_sequences([seed_text])[0]

    predicted_tokens = responser(25, model, tokenized_seed, token_to_idx, idx_to_token, max_seq_len, penalty=1.5)
    print(f"Predicted tokens: {' '.join(predicted_tokens)}")

    with open("response.txt", "w") as f:
        for token in predicted_tokens:
            f.write(" " + token)
