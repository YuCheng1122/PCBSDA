import os

from transformers import (
    RobertaForMaskedLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

def create_model(config):
    model = RobertaForMaskedLM(config)
    print("Model initialized successfully, number of parameters:", sum(p.numel() for p in model.parameters()))
    return model

def init_pretrain_components(config, model, tokenizer, train_dataset, eval_dataset=None):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=config["mlm_probability"]
    )

    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    training_args = TrainingArguments(
        output_dir=config["checkpoint_dir"],
        overwrite_output_dir=True,
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        save_steps=config["save_steps"],
        save_total_limit=config["save_total_limit"],
        logging_dir=f'{config["checkpoint_dir"]}/logs',
        logging_steps=config["logging_steps"],
        prediction_loss_only=True,
        lr_scheduler_type="linear",
        warmup_steps=config["warmup_steps"],
        eval_strategy="epoch" if eval_dataset else "no",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    return trainer
