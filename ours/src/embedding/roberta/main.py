import os
import json
import math
from datasets import load_from_disk
from transformers import RobertaConfig, TrainerCallback

from ours.src.embedding.roberta.utils import load_corpus_dataset, setup_training_environment
from ours.configs.embedding.roberta.pretrain import get_pretrain_config
from ours.src.embedding.roberta.tokenizer import build_vocab_from_dataset, create_wordlevel_tokenizer
from ours.src.embedding.roberta.models import init_pretrain_components, create_model


class LossTrackingCallback(TrainerCallback):
    """Custom callback to track training loss and save model at specific epoch"""

    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.epoch_losses = []
        self.current_epoch_losses = []
        self.current_epoch_lrs = []
        self.loss_log_path = os.path.join(config["checkpoint_dir"], "training_losses.json")
        self.save_epochs = config.get("save_at_epochs", [])

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging happens during training"""
        if logs and "loss" in logs:
            loss = logs["loss"]
            lr = logs.get("learning_rate", None)
            ppl = math.exp(loss) if loss < 100 else float("inf")
            self.current_epoch_losses.append(loss)
            if lr is not None:
                self.current_epoch_lrs.append(lr)
            print(f"Step {state.global_step}: Loss = {loss:.4f} | PPL = {ppl:.2f} | LR = {lr:.2e}" if lr else
                  f"Step {state.global_step}: Loss = {loss:.4f} | PPL = {ppl:.2f}")

        # Log eval loss
        if logs and "eval_loss" in logs:
            eval_loss = logs["eval_loss"]
            eval_ppl = math.exp(eval_loss) if eval_loss < 100 else float("inf")
            print(f"  Eval Loss = {eval_loss:.4f} | Eval PPL = {eval_ppl:.2f}")

    def on_epoch_end(self, args, state, control, **kwargs):
        """Called at the end of each epoch"""
        if self.current_epoch_losses:
            avg_loss = sum(self.current_epoch_losses) / len(self.current_epoch_losses)
            avg_ppl = math.exp(avg_loss) if avg_loss < 100 else float("inf")
            avg_lr = sum(self.current_epoch_lrs) / len(self.current_epoch_lrs) if self.current_epoch_lrs else None

            epoch_record = {
                "epoch": int(state.epoch),
                "avg_loss": avg_loss,
                "avg_perplexity": avg_ppl,
                "avg_lr": avg_lr,
                "step": state.global_step
            }
            self.epoch_losses.append(epoch_record)

            print(f"\n{'='*60}")
            print(f"Epoch {int(state.epoch)} completed")
            print(f"  Avg Loss: {avg_loss:.4f} | Avg PPL: {avg_ppl:.2f} | Avg LR: {avg_lr:.2e}" if avg_lr else
                  f"  Avg Loss: {avg_loss:.4f} | Avg PPL: {avg_ppl:.2f}")
            print(f"{'='*60}\n")

            with open(self.loss_log_path, 'w') as f:
                json.dump(self.epoch_losses, f, indent=2)

            if int(state.epoch) in self.save_epochs:
                save_path = os.path.join(
                    self.config["checkpoint_dir"],
                    f"model_epoch_{int(state.epoch)}"
                )
                os.makedirs(save_path, exist_ok=True)
                kwargs["model"].save_pretrained(save_path)
                self.tokenizer.save_pretrained(save_path)
                print(f"\n{'*'*60}")
                print(f"Model checkpoint saved at epoch {int(state.epoch)}")
                print(f"Saved to: {save_path}")
                print(f"{'*'*60}\n")

            self.current_epoch_losses = []
            self.current_epoch_lrs = []


def main():
    device = setup_training_environment()
    config = get_pretrain_config()

    # Step 1: Load corpus
    raw_dataset = load_corpus_dataset(config["corpus_path"])

    print("\n" + "="*60)
    print("Dataset Info:")
    print(f"Total samples: {len(raw_dataset)}")
    print(f"\nFirst 5 samples:")
    for i in range(min(5, len(raw_dataset))):
        print(f"Sample {i}: {raw_dataset[i]['text'][:100]}...")
    print("="*60)

    # Step 2: Build vocab from corpus and create tokenizer
    vocab = build_vocab_from_dataset(raw_dataset, config["special_tokens"])
    vocab_size = len(vocab)

    tokenizer = create_wordlevel_tokenizer(
        vocab=vocab,
        special_tokens=config["special_tokens"],
        max_length=config["max_length"],
        save_dir=config["tokenizer_output_dir"],
    )

    # Step 3: Build model config with dynamic vocab size
    model_config = RobertaConfig(
        vocab_size=vocab_size,
        max_position_embeddings=config["max_position_embeddings"],
        num_attention_heads=config["num_attention_heads"],
        num_hidden_layers=config["num_hidden_layers"],
        type_vocab_size=config["type_vocab_size"],
        hidden_size=config["hidden_size"],
        intermediate_size=config["intermediate_size"],
    )

    print("\n" + "="*60)
    print("Config:")
    print(f"vocab_size: {vocab_size} (dynamic)")
    for key, value in config.items():
        if key not in ("special_tokens",):
            print(f"{key}: {value}")
    print(f"special_tokens: {config['special_tokens']}")
    print("="*60 + "\n")

    # Step 4: Tokenize dataset
    tokenized_cache_path = config["corpus_path"].replace(".pkl", "_tokenized")

    if os.path.exists(tokenized_cache_path):
        print(f"Loading tokenized dataset from cache: {tokenized_cache_path}")
        tokenized_dataset = load_from_disk(tokenized_cache_path)
        print(f"Loaded {len(tokenized_dataset)} tokenized samples")
    else:
        print("Tokenizing dataset (will be cached)...")

        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True)

        tokenized_dataset = raw_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=24,
            remove_columns=["text"],
            desc="Tokenizing"
        )

        print(f"Saving tokenized dataset to: {tokenized_cache_path}")
        tokenized_dataset.save_to_disk(tokenized_cache_path)
        print("Cache saved!")

    # Step 5: Split train/eval
    eval_ratio = config.get("eval_ratio", 0.05)
    split = tokenized_dataset.train_test_split(test_size=eval_ratio, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"Train: {len(train_dataset)} | Eval: {len(eval_dataset)} ({eval_ratio*100:.0f}%)")

    # Step 6: Create model and train
    model = create_model(model_config)
    model.to(device)

    trainer = init_pretrain_components(config, model, tokenizer, train_dataset, eval_dataset)

    loss_callback = LossTrackingCallback(config, tokenizer)
    trainer.add_callback(loss_callback)
    print(f"Loss tracking enabled - logs will be saved to {loss_callback.loss_log_path}")
    print(f"Model checkpoints will be saved at epochs: {config.get('save_at_epochs', [])}")
    print(f"Warmup steps: {config['warmup_steps']}")
    print(f"LR Scheduler: linear\n")

    print("Starting training...")
    trainer.train()

    # Save final model and tokenizer
    trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])
    print(f"Final model and tokenizer saved to {config['output_dir']}")
    print(f"Training checkpoints saved in {config['checkpoint_dir']}")

if __name__ == "__main__":
    main()
