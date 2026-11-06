import math
import random
import json
import os

def matrix_multiply(A, B):
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    if cols_A != rows_B:
        raise ValueError(f"Dimension mismatch: {cols_A} != {rows_B}")
    result = [[0.0 for _ in range(cols_B)] for _ in range(rows_A)]
    for i in range(rows_A):
        for k in range(cols_A):
            a_ik = A[i][k]
            b_row = B[k]
            res_row = result[i]
            for j in range(cols_B):
                res_row[j] += a_ik * b_row[j]
    return result

def matrix_transpose(A):
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]

def vector_add(v1, v2):
    if isinstance(v1[0], list):
        return [[a + b for a, b in zip(row1, row2)] for row1, row2 in zip(v1, v2)]
    return [a + b for a, b in zip(v1, v2)]

def softmax(vector, temperature=1.0):
    if temperature <= 0:
        temperature = 1e-5
    scaled = [x / temperature for x in vector]
    max_val = max(scaled)
    exp_vals = [math.exp(max(-50, min(50, x - max_val))) for x in scaled]
    sum_exp = sum(exp_vals)
    if sum_exp == 0:
        return [1.0 / len(vector) for _ in vector]
    return [v / sum_exp for v in exp_vals]

def layer_norm(matrix, epsilon=1e-5):
    normalized = []
    for row in matrix:
        mean = sum(row) / len(row)
        variance = sum((x - mean) ** 2 for x in row) / len(row)
        std = math.sqrt(variance + epsilon)
        normalized.append([(x - mean) / std for x in row])
    return normalized

def gelu(x):
    return 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))

class AdamOptimizer:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, param_id, weights, gradients):
        self.t += 1
        if param_id not in self.m:
            self.m[param_id] = [[0.0 for _ in row] for row in weights]
            self.v[param_id] = [[0.0 for _ in row] for row in weights]
            
        for i in range(len(weights)):
            for j in range(len(weights[0])):
                g = gradients[i][j]
                self.m[param_id][i][j] = self.beta1 * self.m[param_id][i][j] + (1 - self.beta1) * g
                self.v[param_id][i][j] = self.beta2 * self.v[param_id][i][j] + (1 - self.beta2) * (g ** 2)
                
                m_hat = self.m[param_id][i][j] / (1 - (self.beta1 ** self.t))
                v_hat = self.v[param_id][i][j] / (1 - (self.beta2 ** self.t))
                
                weights[i][j] -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)

    def state_dict(self):
        return {"t": self.t, "m": self.m, "v": self.v}

    def load_state_dict(self, state, parameters):
        if not isinstance(state, dict) or not isinstance(state.get("t"), int):
            raise ValueError("Invalid optimizer state in checkpoint.")
        moments = state.get("m")
        variances = state.get("v")
        if not isinstance(moments, dict) or not isinstance(variances, dict):
            raise ValueError("Invalid optimizer moments in checkpoint.")

        for param_id, weights in parameters.items():
            for values in (moments.get(param_id), variances.get(param_id)):
                if values is None or len(values) != len(weights):
                    raise ValueError(f"Invalid optimizer state shape for {param_id}.")
                if any(len(row) != len(weight_row) for row, weight_row in zip(values, weights)):
                    raise ValueError(f"Invalid optimizer state shape for {param_id}.")

        self.t = state["t"]
        self.m = moments
        self.v = variances

class Tokenizer:
    def __init__(self, text_corpus):
        chars = sorted(list(set(text_corpus)))
        self.vocab = chars + ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        self.vocab_size = len(self.vocab)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.vocab)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.vocab)}

    def encode(self, text):
        return [self.char_to_idx.get(ch, self.char_to_idx['<UNK>']) for ch in text]

    def decode(self, indices):
        return "".join([self.idx_to_char.get(idx, '<UNK>') for idx in indices if idx < len(self.vocab)])

class EmbeddingLayer:
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        limit = math.sqrt(6.0 / (vocab_size + embedding_dim))
        self.weights = [[random.uniform(-limit, limit) for _ in range(embedding_dim)] for _ in range(vocab_size)]

    def forward(self, token_indices):
        return [self.weights[idx] for idx in token_indices]

class PositionalEncoding:
    def __init__(self, max_len, embedding_dim):
        self.encoding = []
        for pos in range(max_len):
            row = []
            for i in range(embedding_dim):
                val = pos / math.pow(10000, (2 * (i // 2)) / embedding_dim)
                if i % 2 == 0:
                    row.append(math.sin(val))
                else:
                    row.append(math.cos(val))
            self.encoding.append(row)

    def forward(self, seq_len):
        return self.encoding[:seq_len]

class MultiHeadAttention:
    def __init__(self, embed_dim, num_heads):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        limit = math.sqrt(6.0 / (embed_dim + embed_dim))
        self.W_q = [[random.uniform(-limit, limit) for _ in range(embed_dim)] for _ in range(embed_dim)]
        self.W_k = [[random.uniform(-limit, limit) for _ in range(embed_dim)] for _ in range(embed_dim)]
        self.W_v = [[random.uniform(-limit, limit) for _ in range(embed_dim)] for _ in range(embed_dim)]
        self.W_o = [[random.uniform(-limit, limit) for _ in range(embed_dim)] for _ in range(embed_dim)]

    def forward(self, x, mask=True):
        seq_len = len(x)
        Q = matrix_multiply(x, self.W_q)
        K = matrix_multiply(x, self.W_k)
        V = matrix_multiply(x, self.W_v)

        head_outputs = []
        for h in range(self.num_heads):
            start = h * self.head_dim
            end = start + self.head_dim
            
            Q_h = [row[start:end] for row in Q]
            K_h = [row[start:end] for row in K]
            V_h = [row[start:end] for row in V]

            K_h_T = matrix_transpose(K_h)
            scores = matrix_multiply(Q_h, K_h_T)
            scale = 1.0 / math.sqrt(self.head_dim)
            
            scored_matrix = []
            for i, row in enumerate(scores):
                scored_row = []
                for j, val in enumerate(row):
                    val *= scale
                    if mask and j > i:
                        val = -1e9
                    scored_row.append(val)
                scored_matrix.append(softmax(scored_row))

            head_out = matrix_multiply(scored_matrix, V_h)
            head_outputs.append(head_out)

        concatenated = []
        for i in range(seq_len):
            concat_row = []
            for h in range(self.num_heads):
                concat_row.extend(head_outputs[h][i])
            concatenated.append(concat_row)

        return matrix_multiply(concatenated, self.W_o)

class FeedForward:
    def __init__(self, embed_dim, hidden_dim):
        limit1 = math.sqrt(6.0 / (embed_dim + hidden_dim))
        limit2 = math.sqrt(6.0 / (hidden_dim + embed_dim))
        self.W1 = [[random.uniform(-limit1, limit1) for _ in range(hidden_dim)] for _ in range(embed_dim)]
        self.b1 = [0.0 for _ in range(hidden_dim)]
        self.W2 = [[random.uniform(-limit2, limit2) for _ in range(embed_dim)] for _ in range(hidden_dim)]
        self.b2 = [0.0 for _ in range(embed_dim)]

    def forward(self, x):
        hidden = []
        for row in x:
            h_row = []
            for j in range(len(self.b1)):
                val = self.b1[j] + sum(row[i] * self.W1[i][j] for i in range(len(row)))
                h_row.append(gelu(val))
            hidden.append(h_row)

        output = []
        for row in hidden:
            out_row = []
            for j in range(len(self.b2)):
                val = self.b2[j] + sum(row[i] * self.W2[i][j] for i in range(len(row)))
                out_row.append(val)
            output.append(out_row)
        return output

class TransformerBlock:
    def __init__(self, embed_dim, num_heads, hidden_dim):
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.ffn = FeedForward(embed_dim, hidden_dim)

    def forward(self, x):
        norm_x = layer_norm(x)
        attn_out = self.attention.forward(norm_x)
        x = vector_add(x, attn_out)
        
        norm_x2 = layer_norm(x)
        ffn_out = self.ffn.forward(norm_x2)
        x = vector_add(x, ffn_out)
        return x

class MiniGPT:
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers, max_seq_len):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embed = EmbeddingLayer(vocab_size, embed_dim)
        self.pos_embed = PositionalEncoding(max_seq_len, embed_dim)
        self.blocks = [TransformerBlock(embed_dim, num_heads, hidden_dim) for _ in range(num_layers)]
        
        limit = math.sqrt(6.0 / (embed_dim + vocab_size))
        self.head_weights = [[random.uniform(-limit, limit) for _ in range(vocab_size)] for _ in range(embed_dim)]
        self.head_bias = [0.0 for _ in range(vocab_size)]

    def forward_features(self, token_indices):
        seq_len = len(token_indices)
        tok_embeds = self.embed.forward(token_indices)
        pos_embeds = self.pos_embed.forward(seq_len)
        
        x = [vector_add(tok_embeds[i], pos_embeds[i]) for i in range(seq_len)]
        
        for block in self.blocks:
            x = block.forward(x)
            
        return layer_norm(x)

    def project(self, features):
        logits = matrix_multiply(features, self.head_weights)
        
        for i in range(len(features)):
            for j in range(len(self.head_bias)):
                logits[i][j] += self.head_bias[j]
                
        return logits

    def forward(self, token_indices):
        return self.project(self.forward_features(token_indices))

    def state_dict(self):
        return {
            "embed": self.embed.weights,
            "blocks": [
                {
                    "attention": {
                        "W_q": block.attention.W_q,
                        "W_k": block.attention.W_k,
                        "W_v": block.attention.W_v,
                        "W_o": block.attention.W_o,
                    },
                    "ffn": {
                        "W1": block.ffn.W1,
                        "b1": block.ffn.b1,
                        "W2": block.ffn.W2,
                        "b2": block.ffn.b2,
                    },
                }
                for block in self.blocks
            ],
            "head_weights": self.head_weights,
            "head_bias": self.head_bias,
        }

    def load_state_dict(self, state):
        if not isinstance(state, dict):
            raise ValueError("Invalid model state in checkpoint.")

        _copy_matrix("embedding weights", self.embed.weights, state.get("embed"))
        blocks = state.get("blocks")
        if not isinstance(blocks, list) or len(blocks) != len(self.blocks):
            raise ValueError("Checkpoint transformer layer count does not match the model.")

        for index, (block, saved) in enumerate(zip(self.blocks, blocks)):
            if not isinstance(saved, dict):
                raise ValueError(f"Invalid checkpoint data for transformer layer {index}.")
            attention = saved.get("attention", {})
            for name in ("W_q", "W_k", "W_v", "W_o"):
                _copy_matrix(
                    f"layer {index} attention {name}",
                    getattr(block.attention, name),
                    attention.get(name),
                )

            ffn = saved.get("ffn", {})
            _copy_matrix(f"layer {index} feed-forward W1", block.ffn.W1, ffn.get("W1"))
            _copy_vector(f"layer {index} feed-forward b1", block.ffn.b1, ffn.get("b1"))
            _copy_matrix(f"layer {index} feed-forward W2", block.ffn.W2, ffn.get("W2"))
            _copy_vector(f"layer {index} feed-forward b2", block.ffn.b2, ffn.get("b2"))

        _copy_matrix("output weights", self.head_weights, state.get("head_weights"))
        _copy_vector("output bias", self.head_bias, state.get("head_bias"))

def _copy_matrix(name, target, saved):
    if not isinstance(saved, list) or len(saved) != len(target):
        raise ValueError(f"Checkpoint {name} has an incompatible shape.")
    if any(not isinstance(row, list) or len(row) != len(target_row)
           for row, target_row in zip(saved, target)):
        raise ValueError(f"Checkpoint {name} has an incompatible shape.")
    for target_row, saved_row in zip(target, saved):
        target_row[:] = saved_row

def _copy_vector(name, target, saved):
    if not isinstance(saved, list) or len(saved) != len(target):
        raise ValueError(f"Checkpoint {name} has an incompatible shape.")
    target[:] = saved

def cross_entropy_loss(logits, targets):
    total_loss = 0.0
    for logit, target in zip(logits, targets):
        probs = softmax(logit)
        prob_target = max(1e-15, probs[target])
        total_loss += -math.log(prob_target)
    return total_loss / len(targets)

def train_step(model, inputs, targets, optimizer):
    features = model.forward_features(inputs)
    logits = model.project(features)
    loss = cross_entropy_loss(logits, targets)
    gradients = [
        [0.0 for _ in range(model.vocab_size)]
        for _ in range(model.embed_dim + 1)
    ]

    for feature, logit, target in zip(features, logits, targets):
        errors = softmax(logit)
        errors[target] -= 1.0
        for i, value in enumerate(feature):
            for j, error in enumerate(errors):
                gradients[i][j] += value * error / len(targets)
        for j, error in enumerate(errors):
            gradients[model.embed_dim][j] += error / len(targets)

    optimizer.update("lm_head", model.head_weights + [model.head_bias], gradients)
    return loss

def save_checkpoint(path, model, tokenizer, optimizer):
    checkpoint = {
        "version": 1,
        "vocab": tokenizer.vocab,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as checkpoint_file:
        json.dump(checkpoint, checkpoint_file)
    os.replace(temporary_path, path)

def load_checkpoint(path, model, tokenizer, optimizer):
    if not os.path.exists(path):
        return "missing"

    with open(path, "r", encoding="utf-8") as checkpoint_file:
        checkpoint = json.load(checkpoint_file)
    if not isinstance(checkpoint, dict):
        raise ValueError("Invalid checkpoint: expected a JSON object.")

    if checkpoint.get("version") == 1:
        if checkpoint.get("vocab") != tokenizer.vocab:
            raise ValueError("Checkpoint tokenizer vocabulary does not match the training corpus.")
        model.load_state_dict(checkpoint.get("model"))
        optimizer_state = checkpoint.get("optimizer")
        if optimizer_state is not None:
            optimizer.load_state_dict(
                optimizer_state,
                {"lm_head": model.head_weights + [model.head_bias]},
            )
        return "full"

    if "head_weights" in checkpoint and "head_bias" in checkpoint:
        _copy_matrix("output weights", model.head_weights, checkpoint["head_weights"])
        _copy_vector("output bias", model.head_bias, checkpoint["head_bias"])
        return "legacy"

    raise ValueError("Unsupported or incomplete model checkpoint.")

def advanced_sample(logits, temperature=0.8, top_k=5, top_p=0.9):
    probs = softmax(logits, temperature=temperature)
    
    sorted_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    if top_k > 0:
        allowed_indices = set(sorted_indices[:top_k])
        probs = [p if i in allowed_indices else 0.0 for i, p in enumerate(probs)]
        
    sum_p = sum(probs)
    if sum_p > 0:
        probs = [p / sum_p for p in probs]
    else:
        return sorted_indices[0]

    cumulative = 0.0
    nucleus_indices = []
    for idx in sorted(sorted_indices, key=lambda i: probs[i], reverse=True):
        cumulative += probs[idx]
        nucleus_indices.append(idx)
        if cumulative >= top_p:
            break
            
    filtered_probs = [probs[i] if i in nucleus_indices else 0.0 for i in range(len(probs))]
    sum_fp = sum(filtered_probs)
    if sum_fp > 0:
        filtered_probs = [p / sum_fp for p in filtered_probs]
    else:
        return sorted_indices[0]

    r = random.random()
    cum = 0.0
    for idx, p in enumerate(filtered_probs):
        cum += p
        if r <= cum:
            return idx
    return sorted_indices[0]

if __name__ == "__main__":
    print("Initializing Enterprise-Grade Vanilla Python Mini-GPT...")
    
    corpus = "artificial intelligence, deep learning, and transformer architectures power modern language models. " * 40
    tokenizer = Tokenizer(corpus)
    
    vocab_size = tokenizer.vocab_size
    embed_dim = 32
    num_heads = 4
    hidden_dim = 64
    num_layers = 2
    max_seq_len = 32
    
    model = MiniGPT(vocab_size, embed_dim, num_heads, hidden_dim, num_layers, max_seq_len)
    optimizer = AdamOptimizer(lr=0.01)
    checkpoint_path = os.path.join(os.path.dirname(__file__), "minigpt_checkpoint.json")
    checkpoint_status = load_checkpoint(checkpoint_path, model, tokenizer, optimizer)
    if checkpoint_status == "full":
        print("Loaded model and optimizer state from checkpoint.")
    elif checkpoint_status == "legacy":
        print("Loaded legacy output weights; transformer layers will be initialized for this run.")
    
    encoded_corpus = tokenizer.encode(corpus)
    input_sequences = []
    target_sequences = []
    
    for i in range(0, len(encoded_corpus) - max_seq_len, max_seq_len):
        chunk = encoded_corpus[i:i + max_seq_len + 1]
        if len(chunk) < 2:
            continue
        input_sequences.append(chunk[:-1])
        target_sequences.append(chunk[1:])

    print(f"Dataset loaded: {len(input_sequences)} sequences.")
    print("Training model before chat mode...")
    
    epochs = 3
    for epoch in range(epochs):
        epoch_loss = 0.0
        training_data = list(zip(input_sequences, target_sequences))
        random.shuffle(training_data)
        for inputs, targets in training_data:
            epoch_loss += train_step(model, inputs, targets, optimizer)

        save_checkpoint(checkpoint_path, model, tokenizer, optimizer)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss / len(input_sequences):.4f}")

    print("\n" + "="*50)
    print("Interactive Chat Mode Ready! (Type 'exit' to quit)")
    print("="*50)

    while True:
        try:
            user_input = input("\nUser Prompt: ")
        except EOFError:
            print("\nInput closed. Model checkpoint is saved.")
            break
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        if not user_input.strip():
            continue
            
        tokens = tokenizer.encode(user_input)
        generation_length = 60
        
        for _ in range(generation_length):
            context = tokens[-max_seq_len:]
            logits = model.forward(context)
            next_token_logits = logits[-1]
            
            next_token = advanced_sample(next_token_logits, temperature=0.7, top_k=6, top_p=0.85)
            tokens.append(next_token)
            
        generated_response = tokenizer.decode(tokens)
        print(f"Mini-GPT: {generated_response}")