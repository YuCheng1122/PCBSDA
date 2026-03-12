def get_pretrain_config():
    MAX_LENGTH = 512
    BASE_PATH = "/home/tommy/Project/PCBSDA"

    config = {
        # Model architecture (vocab_size and model_config will be set dynamically)
        "max_position_embeddings": MAX_LENGTH + 2,
        "num_attention_heads": 8,
        "num_hidden_layers": 6,
        "type_vocab_size": 1,
        "hidden_size": 256,
        "intermediate_size": 1024,

        # Data paths
        "corpus_path": f"{BASE_PATH}/ours/outputs/embedding/corpus_Advanced Micro Devices x86-64.pkl",
        "output_dir": f"{BASE_PATH}/ours/outputs/models/embedding/roberta/",
        "tokenizer_output_dir": f"{BASE_PATH}/ours/outputs/tokenizer/roberta/",
        "checkpoint_dir": f"{BASE_PATH}/ours/outputs/checkpoints/embedding/roberta/",

        # Training parameters
        "epochs": 100,
        "batch_size": 64,
        "learning_rate": 1e-4,
        "warmup_steps": 10000,
        "max_length": MAX_LENGTH,
        "eval_ratio": 0.05,
        "save_at_epochs": [10, 20, 30, 50, 75, 100],

        # MLM parameters
        "mlm_probability": 0.15,

        # Tokenizer parameters
        "special_tokens": {
            "unk_token": "[UNK]",
            "pad_token": "[PAD]",
            "cls_token": "[CLS]",
            "sep_token": "[SEP]",
            "mask_token": "[MASK]"
        },

        # Saving parameters
        "save_steps": 10000,
        "save_total_limit": 2,
        "logging_steps": 100
    }
    return config
