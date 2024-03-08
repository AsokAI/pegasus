import numpy as np
import datasets
from transformers import TrainingArguments, Trainer
import pegasusai as peg
import pegasusai.datasets as peg_datasets
import pegasusai.transformers as peg_transformers
import pegasusai.evaluate as peg_evaluate


dataset = datasets.load_dataset(path="yelp_review_full")
tokenizer = peg_transformers.loading.load_tokenizer("bert-base-cased")

tokenized_datasets = peg_transformers.models.bert_tokenizer_fast_tokenize_dataset_dict(
    input_tokenizer=tokenizer,
    input_dataset_dict=dataset,
    tokenizer_target_column="text",
    padding="max_length",
    truncation=True,
    batched=True,
)

small_train_dataset = peg.commons.getitem(tokenized_datasets, "train")
small_train_dataset = peg_datasets.dataset_shuffle(small_train_dataset, seed=42)
small_train_dataset = peg_datasets.dataset_select(small_train_dataset, range(1000))

small_eval_dataset = peg.commons.getitem(tokenized_datasets, "test")
small_eval_dataset = peg_datasets.dataset_shuffle(small_eval_dataset, seed=42)
small_eval_dataset = peg_datasets.dataset_select(small_eval_dataset, range(100))

model = peg_transformers.loading.load_model(
    "bert-base-cased", peg_transformers.Task.SequenceClassification, num_labels=5
)

logit_transform = peg_transformers.trainer.Transform(
    function="numpy.argmax", kwargs={"axis": -1}
)
metric_config = peg_transformers.trainer.MetricConfig(
    metric_name="accuracy",
    logit_transforms=[logit_transform],
    label_transforms=[]
)

training_args = TrainingArguments(output_dir="test_trainer", num_train_epochs=1)
trainer = peg_transformers.trainer.trainer___init__(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    metric_config=metric_config,
)
peg_transformers.trainer.trainer_train(trainer)
