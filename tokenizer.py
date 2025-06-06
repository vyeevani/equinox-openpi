import sentencepiece
import numpy as np
import logging
import urllib.request
import pathlib

class Tokenizer:
    def __init__(self, max_len: int = 48):
        self._max_len = max_len
        path = pathlib.Path("paligemma_tokenizer.model")
        if not path.exists():
            urllib.request.urlretrieve("https://storage.googleapis.com/big_vision/paligemma_tokenizer.model", path)
        with path.open("rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

    def tokenize(self, prompt: str) -> tuple[np.ndarray, np.ndarray]:
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        # tokenize "\n" separately as the "start of answer" token
        tokens = self._tokenizer.encode(cleaned_text, add_bos=True) + self._tokenizer.encode("\n")
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            mask = [True] * tokens_len + padding
            tokens = tokens + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len
        return np.asarray(tokens), np.asarray(mask)
    def detokenize(self, tokens: np.ndarray) -> str:
        """Convert tokens back to text.
        
        Args:
            tokens: Array of token IDs to convert to text
            
        Returns:
            Decoded text string
        """
        # Remove padding tokens (0) and decode
        tokens = tokens[tokens != 0]
        return self._tokenizer.decode(tokens.tolist())
    
if __name__ == "__main__":
    tokenizer = Tokenizer()
    tokens, mask = tokenizer.tokenize("tell me a joke")
    print(tokens)
    print(mask)
    print(tokenizer.detokenize(tokens))