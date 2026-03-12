from tokenizers import Tokenizer, pre_tokenizers, processors
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast
import os


def build_vocab_from_dataset(dataset, special_tokens):
    """Scan dataset to collect unique tokens and build vocab dict."""
    special_token_list = [
        special_tokens["unk_token"],
        special_tokens["pad_token"],
        special_tokens["cls_token"],
        special_tokens["sep_token"],
        special_tokens["mask_token"],
    ]

    unique_tokens = set()
    for example in dataset:
        unique_tokens.update(example["text"].split())

    # Special tokens first, then sorted corpus tokens
    vocab = {tok: i for i, tok in enumerate(special_token_list)}
    for tok in sorted(unique_tokens):
        if tok not in vocab:
            vocab[tok] = len(vocab)

    print(f"Vocab built: {len(vocab)} tokens ({len(special_token_list)} special + {len(vocab) - len(special_token_list)} corpus tokens)")
    return vocab


def create_wordlevel_tokenizer(vocab, special_tokens, max_length, save_dir=None):
    """
    Create a WordLevel tokenizer from a vocab dict.

    Args:
        vocab: dict mapping token -> id
        special_tokens: dict with unk_token, pad_token, cls_token, sep_token, mask_token
        max_length: max sequence length
        save_dir: optional directory to save tokenizer

    Returns:
        PreTrainedTokenizerFast
    """
    tokenizer = Tokenizer(WordLevel(vocab, unk_token=special_tokens["unk_token"]))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{special_tokens['cls_token']} $A {special_tokens['sep_token']}",
        special_tokens=[
            (special_tokens["cls_token"], vocab[special_tokens["cls_token"]]),
            (special_tokens["sep_token"], vocab[special_tokens["sep_token"]]),
        ],
    )

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        model_max_length=max_length,
        **special_tokens,
    )

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fast_tokenizer.save_pretrained(save_dir)
        print(f"Tokenizer saved to {save_dir}")

    return fast_tokenizer
