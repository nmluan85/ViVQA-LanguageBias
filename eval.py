from transformers import TrainingArguments, Trainer, EarlyStoppingCallback, AutoTokenizer
import transformers
import mlflow
import argparse
import numpy as np
import proposed_model
import torch
from timm.models import create_model
from sklearn.metrics import accuracy_score
from dataset_css import get_dataset
from css_trainer import CSSTrainer

def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    return {"accuracy": accuracy}

def get_options():
    args = argparse.ArgumentParser()

    # Training Argument
    args.add_argument("--output-dir", type=str, default="./output")
    args.add_argument("--log-level", choices=["debug", "info", "warning", "error", "critical", "passive"], default="passive")
    args.add_argument("--lr-scheduler-type", choices=["cosine", "linear"], default="cosine")
    args.add_argument("--warmup-ratio", type=float, default=0.1)
    args.add_argument("--logging-strategy", choices=["no", "epoch", "steps"], default="epoch")
    args.add_argument("--save-strategy", choices=["no", "epoch", "steps"], default="epoch")
    args.add_argument("--save-total-limit", type=int, default=1)
    args.add_argument("-tb", "--train-batch-size", type=int, default=8)
    args.add_argument("--gradient-accumulation-steps", type=int, default=4)
    args.add_argument("-eb", "--eval-batch-size", type=int, default=32)
    args.add_argument("-e", "--epochs", type=int, default=20)
    args.add_argument("-lr", "--learning-rate", type=float, default=3e-5)
    args.add_argument("--weight-decay", type=float, default=0.01)
    args.add_argument("--workers", type=int, default=0)

    # Varriables setting
    args.add_argument("--image-path", type=str, default="./data/images")
    args.add_argument("--ans-path", type=str, default="./data/vocab.json")
    args.add_argument("--train-path", type=str, default="./data/ViVQA-csv/train.csv")
    args.add_argument("--val-path", type=str, default="./data/ViVQA-csv/val.csv")
    args.add_argument("--test-path", type=str, default="./data/ViVQA-csv/test.csv")
    args.add_argument("--ckpt-path", type=str, default="./output/pytorch_model.bin")
    args.add_argument("--feature-paths", type=str, default="./features")
    args.add_argument("--css-candidates", type=str, default="./css-candidates")

    # VinVL parameter
    args.add_argument("--top-K", type=int, default=36)
    args.add_argument("--top-k", type=int, default=10)

    # CSS parameter
    args.add_argument("--use-vcss", type=bool, default=True)
    args.add_argument("--use-qcss", type=bool, default=True)
    args.add_argument("--css-types", nargs="+", choices=["counting", "color"], default=["counting", "color"])
    args.add_argument("--css-interval", type=int, default=2)
    args.add_argument("--mask-value", type=float, default=0.0)
    args.add_argument("--alpha-css", type=float, default=0.1)
    args.add_argument("--css-start-step", type=int, default=500)

    # Model setting
    args.add_argument("--drop-path-rate", type=float, default=0.3)
    args.add_argument("--encoder-layers", type=int, default=6)
    args.add_argument("--encoder-attention-heads-layers", type=int, default=6)
    args.add_argument("--classes", type=int, default=353)

    opt = args.parse_args()
    return opt

def _get_train_config(opt):
    args = TrainingArguments(
        output_dir=opt.output_dir,
        log_level=opt.log_level,
        lr_scheduler_type=opt.lr_scheduler_type,
        warmup_ratio=opt.warmup_ratio,
        logging_strategy=opt.logging_strategy,
        save_strategy=opt.save_strategy,
        save_total_limit=opt.save_total_limit,
        per_device_train_batch_size=opt.train_batch_size,
        per_device_eval_batch_size=opt.eval_batch_size,
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
        num_train_epochs=opt.epochs,
        learning_rate=opt.learning_rate,
        weight_decay=opt.weight_decay,
        dataloader_num_workers=opt.workers,
        report_to='none',
        # report_to='mlflow',
        save_safetensors=False,
        disable_tqdm=False,
        logging_steps=10,
        overwrite_output_dir=True,
        metric_for_best_model='accuracy',
        eval_strategy='epoch',
        load_best_model_at_end=True,
        greater_is_better=True,
        remove_unused_columns=False
    )
    return args

def main():
    opt = get_options()
    model = create_model('proposed_vivqa_model', 
                        num_classes=opt.classes, 
                        drop_path_rate=opt.drop_path_rate,
                        encoder_layers=opt.encoder_layers,
                        encoder_attention_heads=opt.encoder_attention_heads_layers)
            
    args = _get_train_config(opt)
    tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-syllable", use_fast=False)

    train_dataset, val_dataset, test_dataset, COUNTING_ANSWER_IDS, COLOR_ANSWER_IDS = get_dataset(opt, tokenizer)


    state_dict = torch.load(opt.ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    trainer = CSSTrainer(
        opt=opt,
        tokenizer=tokenizer,
        COUNTING_ANSWER_IDS=COUNTING_ANSWER_IDS,
        COLOR_ANSWER_IDS=COLOR_ANSWER_IDS,
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    test_results = trainer.evaluate(test_dataset)
    print(test_results)

if __name__ == '__main__':
    main()