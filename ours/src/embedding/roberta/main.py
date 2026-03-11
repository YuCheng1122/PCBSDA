import os
import json
from datasets import load_from_disk
from transformers import TrainerCallback

from ours.src.embedding.roberta.utils import load_corpus_dataset, setup_training_environment
from ours.configs.embedding.roberta.pretrain import get_pretrain_config
from ours.src.embedding.roberta.tokenizer import create_wordlevel_tokenizer
from ours.src.embedding.roberta.models import init_pretrain_components, create_model


class LossTrackingCallback(TrainerCallback):
    """Custom callback to track training loss and save model at specific epoch"""
    
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.epoch_losses = []
        self.current_epoch_losses = []
        self.loss_log_path = os.path.join(config["checkpoint_dir"], "training_losses.json")
        self.save_epochs = config.get("save_at_epochs", [])
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging happens during training"""
        if logs and "loss" in logs:
            self.current_epoch_losses.append(logs["loss"])
            # Print loss every logging step
            print(f"Step {state.global_step}: Loss = {logs['loss']:.4f}")
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Called at the end of each epoch"""
        if self.current_epoch_losses:
            avg_loss = sum(self.current_epoch_losses) / len(self.current_epoch_losses)
            self.epoch_losses.append({
                "epoch": int(state.epoch),
                "avg_loss": avg_loss,
                "step": state.global_step
            })
            print(f"\n{'='*60}")
            print(f"Epoch {int(state.epoch)} completed - Average Loss: {avg_loss:.4f}")
            print(f"{'='*60}\n")
            
            # Save loss history to JSON file
            with open(self.loss_log_path, 'w') as f:
                json.dump(self.epoch_losses, f, indent=2)
            
            # Save model at specified epoch
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
            
            # Clear current epoch losses for next epoch
            self.current_epoch_losses = []


def main():
    # Setup training environment with GPU support
    device = setup_training_environment()
    
    # Load configuration
    config = get_pretrain_config()
    
    # Setup tokenizer
    tokenizer = create_wordlevel_tokenizer(
        vocab_path=config["vocab_path"],
        output_path=config["tokenizer_output_path"],
        special_tokens=config["special_tokens"],
        max_length=config["max_length"]
    )

    raw_dataset = load_corpus_dataset(config["corpus_path"])
    
    print("\n" + "="*60)
    print("Dataset Info:")
    print(f"Total samples: {len(raw_dataset)}")
    print(f"\nFirst 5 samples:")
    for i in range(min(5, len(raw_dataset))):
        print(f"Sample {i}: {raw_dataset[i]['text'][:100]}...")
    print("="*60)
    
    print("\n" + "="*60)
    print("Config:")
    for key, value in config.items():
        if key != "model_config" and key != "special_tokens":
            print(f"{key}: {value}")
    print(f"special_tokens: {config['special_tokens']}")
    print("="*60 + "\n")
    
    tokenized_cache_path = config["corpus_path"].replace(".pkl", "_tokenized")
    
    if os.path.exists(tokenized_cache_path):
        print(f"Loading tokenized dataset from cache: {tokenized_cache_path}")
        tokenized_dataset = load_from_disk(tokenized_cache_path)
        print(f"Loaded {len(tokenized_dataset)} tokenized samples")
    else:
        print("Tokenizing dataset (will be cached)...")
        
        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, padding="max_length")

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

    # Create model and move to device
    model = create_model(config["model_config"])
    model.to(device)
    
    # Initialize trainer components
    trainer = init_pretrain_components(config, model, tokenizer, tokenized_dataset)
    
    # Add loss tracking callback
    loss_callback = LossTrackingCallback(config, tokenizer)
    trainer.add_callback(loss_callback)
    print(f"Loss tracking enabled - logs will be saved to {loss_callback.loss_log_path}")
    print(f"Model checkpoints will be saved at epochs: {config.get('save_at_epochs', 'None')}")
    print(f"LR Scheduler: linear with {config.get('warmup_steps', 10000)} warmup steps (configured in TrainingArguments)\n")

    print("Starting training...")
    trainer.train()

    # Save final model and tokenizer to output directory
    trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])
    print(f"Final model and tokenizer saved to {config['output_dir']}")
    print(f"Training checkpoints saved in {config['checkpoint_dir']}")

if __name__ == "__main__":
    main()
