from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Split, BertPreTokenizer
from tokenizers.processors import TemplateProcessing
from tokenizers.normalizers import NFD, Lowercase, StripAccents, BertNormalizer
from tokenizers.decoders import BPEDecoder


class MyTokenizer:
    def __init__(
        self,
        path: str = None,
        *,
        unk_token: str = "[UNK]",
        end_of_word_suffix: str = "##",
        special_tokens: list = ["[PAD]", "[UNK]", "[SOS]", "[EOS]"],
        vocab_size: int = 2048,
        start_token: tuple = ("[SOS]", 2),
        end_token: tuple = ("[EOS]", 3),
    ):
        if path:
            self.load(path)
        else:
            self.tokenizer = Tokenizer(
                BPE(unk_token=unk_token, end_of_word_suffix=end_of_word_suffix)
            )
            self.trainer = BpeTrainer(
                special_tokens=special_tokens,
                vocab_size=vocab_size,
                show_progress=True,
                end_of_word_suffix=end_of_word_suffix,
            )
            self.tokenizer.pre_tokenizer = BertPreTokenizer()
            self.tokenizer.normalizer = BertNormalizer(lowercase=False)
            self.tokenizer.post_processor = TemplateProcessing(
                single=f"{start_token[0]} $A {end_token[0]}",
                special_tokens=[start_token, end_token],
            )
            self.tokenizer.decoder = BPEDecoder(suffix=end_of_word_suffix)

    def train(self, iterator: list):
        self.tokenizer.train_from_iterator(iterator, self.trainer)
        self.save("tokenizer/tokenizer.json")

    def save(self, path: str):
        self.tokenizer.save(path)

    def load(self, path: str):
        self.tokenizer = Tokenizer.from_file(path)

    def encode(self, text: str):
        return self.tokenizer.encode(text)

    def decode(self, ids: list):
        return self.tokenizer.decode(ids)

    def get_tokenizer(self):
        return self.tokenizer
