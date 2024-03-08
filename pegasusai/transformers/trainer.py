import importlib
from functools import partial
from types import ModuleType
from typing import Any, Callable, Dict, ForwardRef, List, Optional, Tuple, Union

import datasets
import torch
import evaluate
import transformers
from pydantic import BaseModel
from transformers import Trainer


class Transform(BaseModel):
    function: str
    kwargs: Dict[str, Any]


class MetricConfig(BaseModel):
    metric_name: str
    logit_transforms: List[Transform]
    label_transforms: List[Transform]


def get_callable(transform: Transform) -> Tuple[ModuleType, Callable]:
    parent_module_name = transform.function.split(".")[0]
    parent_module = importlib.import_module(parent_module_name)
    function = eval(transform.function)
    partial_function_with_kwargs = partial(function, **transform.kwargs)
    return parent_module, partial_function_with_kwargs


class Evaluator:
    def __init__(self, metric_config: MetricConfig):
        self.metric = evaluate.load(metric_config.metric_name)
        self.logit_transforms = [
            get_callable(transform) for transform in metric_config.logit_transforms
        ]
        self.label_transforms = [
            get_callable(transform) for transform in metric_config.label_transforms
        ]

    def compute(self, eval_pred):
        logits, labels = eval_pred
        for transform in self.logit_transforms:
            logits = transform(logits)
        for transform in self.label_transforms:
            labels = transform(labels)
        return self.metric.compute(predictions=logits, references=labels)


def trainer___init__(
    trainer: Trainer,
    model: Union[
        transformers.modeling_utils.PreTrainedModel, torch.nn.modules.module.Module
    ] = None,
    args: transformers.training_args.TrainingArguments = None,
    data_collator: Optional[transformers.data.data_collator.DataCollator] = None,
    train_dataset: Optional[torch.utils.data.dataset.Dataset] = None,
    eval_dataset: Union[
        torch.utils.data.dataset.Dataset,
        Dict[str, torch.utils.data.dataset.Dataset],
        None,
    ] = None,
    tokenizer: Optional[
        transformers.tokenization_utils_base.PreTrainedTokenizerBase
    ] = None,
    model_init: Optional[
        Callable[[], transformers.modeling_utils.PreTrainedModel]
    ] = None,
    metric_config: Optional[MetricConfig] = None,
    callbacks: Optional[List[transformers.trainer_callback.TrainerCallback]] = None,
    optimizers: Tuple[
        torch.optim.optimizer.Optimizer, torch.optim.lr_scheduler.LambdaLR
    ] = (None, None),
    preprocess_logits_for_metrics: Optional[
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ] = None,
) -> None:
    evaluator = Evaluator(metric_config)
    return trainer.__init__(
        model=model,
        args=args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        model_init=model_init,
        compute_metrics=evaluator.compute,
        callbacks=callbacks,
        optimizers=optimizers,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )


def trainer__activate_neftune(trainer: Trainer, model) -> None:
    return trainer._activate_neftune(
        model=model,
    )


def trainer__add_sm_patterns_to_gitignore(trainer: Trainer) -> None:
    return trainer._add_sm_patterns_to_gitignore()


def trainer__deactivate_neftune(trainer: Trainer, model) -> None:
    return trainer._deactivate_neftune(
        model=model,
    )


def trainer__finish_current_push(trainer: Trainer) -> None:
    return trainer._finish_current_push()


def trainer__gather_and_numpify(trainer: Trainer, tensors, name) -> None:
    return trainer._gather_and_numpify(
        tensors=tensors,
        name=name,
    )


def trainer__get_collator_with_removed_columns(
    trainer: Trainer, data_collator: Callable, description: Optional[str] = None
) -> Callable:
    return trainer._get_collator_with_removed_columns(
        data_collator=data_collator,
        description=description,
    )


def trainer__get_eval_sampler(
    trainer: Trainer, eval_dataset: torch.utils.data.dataset.Dataset
) -> Optional[torch.utils.data.sampler.Sampler]:
    return trainer._get_eval_sampler(
        eval_dataset=eval_dataset,
    )


def trainer__get_learning_rate(trainer: Trainer) -> None:
    return trainer._get_learning_rate()


def trainer__get_output_dir(trainer: Trainer, trial) -> None:
    return trainer._get_output_dir(
        trial=trial,
    )


def trainer__get_train_sampler(
    trainer: Trainer,
) -> Optional[torch.utils.data.sampler.Sampler]:
    return trainer._get_train_sampler()


def trainer__hp_search_setup(trainer: Trainer, trial: Union[ForwardRef("optuna.Trial"), Dict[str, Any]]) -> None:  # type: ignore
    return trainer._hp_search_setup(
        trial=trial,
    )


def trainer__inner_training_loop(
    trainer: Trainer,
    batch_size=None,
    args=None,
    resume_from_checkpoint=None,
    trial=None,
    ignore_keys_for_eval=None,
) -> None:
    return trainer._inner_training_loop(
        batch_size=batch_size,
        args=args,
        resume_from_checkpoint=resume_from_checkpoint,
        trial=trial,
        ignore_keys_for_eval=ignore_keys_for_eval,
    )


def trainer__issue_warnings_after_load(trainer: Trainer, load_result) -> None:
    return trainer._issue_warnings_after_load(
        load_result=load_result,
    )


def trainer__load_best_model(trainer: Trainer) -> None:
    return trainer._load_best_model()


def trainer__load_from_checkpoint(
    trainer: Trainer, resume_from_checkpoint, model=None
) -> None:
    return trainer._load_from_checkpoint(
        resume_from_checkpoint=resume_from_checkpoint,
        model=model,
    )


def trainer__load_optimizer_and_scheduler(trainer: Trainer, checkpoint) -> None:
    return trainer._load_optimizer_and_scheduler(
        checkpoint=checkpoint,
    )


def trainer__load_rng_state(trainer: Trainer, checkpoint) -> None:
    return trainer._load_rng_state(
        checkpoint=checkpoint,
    )


def trainer__maybe_log_save_evaluate(
    trainer: Trainer, tr_loss, model, trial, epoch, ignore_keys_for_eval
) -> None:
    return trainer._maybe_log_save_evaluate(
        tr_loss=tr_loss,
        model=model,
        trial=trial,
        epoch=epoch,
        ignore_keys_for_eval=ignore_keys_for_eval,
    )


def trainer__move_model_to_device(trainer: Trainer, model, device) -> None:
    return trainer._move_model_to_device(
        model=model,
        device=device,
    )


def trainer__nested_gather(trainer: Trainer, tensors, name=None) -> None:
    return trainer._nested_gather(
        tensors=tensors,
        name=name,
    )


def trainer__prepare_input(
    trainer: Trainer, data: Union[torch.Tensor, Any]
) -> Union[torch.Tensor, Any]:
    return trainer._prepare_input(
        data=data,
    )


def trainer__prepare_inputs(
    trainer: Trainer, inputs: Dict[str, Union[torch.Tensor, Any]]
) -> Dict[str, Union[torch.Tensor, Any]]:
    return trainer._prepare_inputs(
        inputs=inputs,
    )


def trainer__push_from_checkpoint(trainer: Trainer, checkpoint_folder) -> None:
    return trainer._push_from_checkpoint(
        checkpoint_folder=checkpoint_folder,
    )


def trainer__remove_unused_columns(
    trainer: Trainer, dataset: "datasets.Dataset", description: Optional[str] = None
) -> None:
    return trainer._remove_unused_columns(
        dataset=dataset,
        description=description,
    )


def trainer__report_to_hp_search(trainer: Trainer, trial: Union[ForwardRef("optuna.Trial"), Dict[str, Any]], step: int, metrics: Dict[str, float]) -> None:  # type: ignore
    return trainer._report_to_hp_search(
        trial=trial,
        step=step,
        metrics=metrics,
    )


def trainer__rotate_checkpoints(
    trainer: Trainer, use_mtime=False, output_dir=None
) -> None:
    return trainer._rotate_checkpoints(
        use_mtime=use_mtime,
        output_dir=output_dir,
    )


def trainer__save(
    trainer: Trainer, output_dir: Optional[str] = None, state_dict=None
) -> None:
    return trainer._save(
        output_dir=output_dir,
        state_dict=state_dict,
    )


def trainer__save_checkpoint(trainer: Trainer, model, trial, metrics=None) -> None:
    return trainer._save_checkpoint(
        model=model,
        trial=trial,
        metrics=metrics,
    )


def trainer__save_optimizer_and_scheduler(trainer: Trainer, output_dir) -> None:
    return trainer._save_optimizer_and_scheduler(
        output_dir=output_dir,
    )


def trainer__save_rng_state(trainer: Trainer, output_dir) -> None:
    return trainer._save_rng_state(
        output_dir=output_dir,
    )


def trainer__save_tpu(trainer: Trainer, output_dir: Optional[str] = None) -> None:
    return trainer._save_tpu(
        output_dir=output_dir,
    )


def trainer__set_signature_columns_if_needed(trainer: Trainer) -> None:
    return trainer._set_signature_columns_if_needed()


def trainer__sorted_checkpoints(
    trainer: Trainer, output_dir=None, checkpoint_prefix="checkpoint", use_mtime=False
) -> List[str]:
    return trainer._sorted_checkpoints(
        output_dir=output_dir,
        checkpoint_prefix=checkpoint_prefix,
        use_mtime=use_mtime,
    )


def trainer__tune_save_checkpoint(trainer: Trainer, checkpoint_dir: str) -> None:
    return trainer._tune_save_checkpoint(
        checkpoint_dir=checkpoint_dir,
    )


def trainer__wrap_model(
    trainer: Trainer, model, training=True, dataloader=None
) -> None:
    return trainer._wrap_model(
        model=model,
        training=training,
        dataloader=dataloader,
    )


def trainer_add_callback(trainer: Trainer, callback) -> None:
    return trainer.add_callback(
        callback=callback,
    )


def trainer_autocast_smart_context_manager(
    trainer: Trainer, cache_enabled: Optional[bool] = True
) -> None:
    return trainer.autocast_smart_context_manager(
        cache_enabled=cache_enabled,
    )


def trainer_call_model_init(trainer: Trainer, trial=None) -> None:
    return trainer.call_model_init(
        trial=trial,
    )


def trainer_compute_loss(trainer: Trainer, model, inputs, return_outputs=False) -> None:
    return trainer.compute_loss(
        model=model,
        inputs=inputs,
        return_outputs=return_outputs,
    )


def trainer_compute_loss_context_manager(trainer: Trainer) -> None:
    return trainer.compute_loss_context_manager()


def trainer_create_accelerator_and_postprocess(trainer: Trainer) -> None:
    return trainer.create_accelerator_and_postprocess()


def trainer_create_model_card(
    trainer: Trainer,
    language: Optional[str] = None,
    license: Optional[str] = None,
    tags: Union[str, List[str], None] = None,
    model_name: Optional[str] = None,
    finetuned_from: Optional[str] = None,
    tasks: Union[str, List[str], None] = None,
    dataset_tags: Union[str, List[str], None] = None,
    dataset: Union[str, List[str], None] = None,
    dataset_args: Union[str, List[str], None] = None,
) -> None:
    return trainer.create_model_card(
        language=language,
        license=license,
        tags=tags,
        model_name=model_name,
        finetuned_from=finetuned_from,
        tasks=tasks,
        dataset_tags=dataset_tags,
        dataset=dataset,
        dataset_args=dataset_args,
    )


def trainer_create_optimizer(trainer: Trainer) -> None:
    return trainer.create_optimizer()


def trainer_create_optimizer_and_scheduler(
    trainer: Trainer, num_training_steps: int
) -> None:
    return trainer.create_optimizer_and_scheduler(
        num_training_steps=num_training_steps,
    )


def trainer_create_scheduler(
    trainer: Trainer,
    num_training_steps: int,
    optimizer: torch.optim.optimizer.Optimizer = None,
) -> None:
    return trainer.create_scheduler(
        num_training_steps=num_training_steps,
        optimizer=optimizer,
    )


def trainer_evaluate(
    trainer: Trainer,
    eval_dataset: Union[
        torch.utils.data.dataset.Dataset,
        Dict[str, torch.utils.data.dataset.Dataset],
        None,
    ] = None,
    ignore_keys: Optional[List[str]] = None,
    metric_key_prefix: str = "eval",
) -> Dict[str, float]:
    return trainer.evaluate(
        eval_dataset=eval_dataset,
        ignore_keys=ignore_keys,
        metric_key_prefix=metric_key_prefix,
    )


def trainer_evaluation_loop(
    trainer: Trainer,
    dataloader: torch.utils.data.dataloader.DataLoader,
    description: str,
    prediction_loss_only: Optional[bool] = None,
    ignore_keys: Optional[List[str]] = None,
    metric_key_prefix: str = "eval",
) -> transformers.trainer_utils.EvalLoopOutput:
    return trainer.evaluation_loop(
        dataloader=dataloader,
        description=description,
        prediction_loss_only=prediction_loss_only,
        ignore_keys=ignore_keys,
        metric_key_prefix=metric_key_prefix,
    )


def trainer_floating_point_ops(
    trainer: Trainer, inputs: Dict[str, Union[torch.Tensor, Any]]
) -> None:
    return trainer.floating_point_ops(
        inputs=inputs,
    )


def trainer_get_decay_parameter_names(trainer: Trainer, model) -> List[str]:
    return trainer.get_decay_parameter_names(
        model=model,
    )


def trainer_get_eval_dataloader(
    trainer: Trainer, eval_dataset: Optional[torch.utils.data.dataset.Dataset] = None
) -> torch.utils.data.dataloader.DataLoader:
    return trainer.get_eval_dataloader(
        eval_dataset=eval_dataset,
    )


def trainer_get_optimizer_cls_and_kwargs(
    trainer: Trainer,
    args: transformers.training_args.TrainingArguments,
):
    return trainer.get_optimizer_cls_and_kwargs(
        args=args,
    )


def trainer_get_test_dataloader(
    trainer: Trainer, test_dataset: torch.utils.data.dataset.Dataset
) -> torch.utils.data.dataloader.DataLoader:
    return trainer.get_test_dataloader(
        test_dataset=test_dataset,
    )


def trainer_get_train_dataloader(
    trainer: Trainer,
) -> torch.utils.data.dataloader.DataLoader:
    return trainer.get_train_dataloader()


def trainer_hyperparameter_search(trainer: Trainer, hp_space: Optional[Callable[[ForwardRef("optuna.Trial")], Dict[str, float]]] = None, compute_objective: Optional[Callable[[Dict[str, float]], float]] = None, n_trials: int = 20, direction: Union[str, List[str]] = "minimize", backend: Union[ForwardRef("str"), transformers.trainer_utils.HPSearchBackend, None] = None, hp_name: Optional[Callable[[ForwardRef("optuna.Trial")], str]] = None, **kwargs) -> Union[transformers.trainer_utils.BestRun, List[transformers.trainer_utils.BestRun]]:  # type: ignore
    return trainer.hyperparameter_search(
        hp_space=hp_space,
        compute_objective=compute_objective,
        n_trials=n_trials,
        direction=direction,
        backend=backend,
        hp_name=hp_name,
        kwargs=kwargs,
    )


def trainer_init_hf_repo(trainer: Trainer) -> None:
    return trainer.init_hf_repo()


def trainer_ipex_optimize_model(
    trainer: Trainer, model, training=False, dtype=torch.float32
) -> None:
    return trainer.ipex_optimize_model(
        model=model,
        training=training,
        dtype=dtype,
    )


def trainer_is_local_process_zero(trainer: Trainer) -> bool:
    return trainer.is_local_process_zero()


def trainer_is_world_process_zero(trainer: Trainer) -> bool:
    return trainer.is_world_process_zero()


def trainer_log(trainer: Trainer, logs: Dict[str, float]) -> None:
    return trainer.log(
        logs=logs,
    )


def trainer_log_metrics(trainer: Trainer, split, metrics) -> None:
    return trainer.log_metrics(
        split=split,
        metrics=metrics,
    )


def trainer_metrics_format(
    trainer: Trainer, metrics: Dict[str, float]
) -> Dict[str, float]:
    return trainer.metrics_format(
        metrics=metrics,
    )


def trainer_num_examples(
    trainer: Trainer, dataloader: torch.utils.data.dataloader.DataLoader
) -> int:
    return trainer.num_examples(
        dataloader=dataloader,
    )


def trainer_num_tokens(
    trainer: Trainer,
    train_dl: torch.utils.data.dataloader.DataLoader,
    max_steps: Optional[int] = None,
) -> int:
    return trainer.num_tokens(
        train_dl=train_dl,
        max_steps=max_steps,
    )


def trainer_pop_callback(trainer: Trainer, callback) -> None:
    return trainer.pop_callback(
        callback=callback,
    )


def trainer_predict(
    trainer: Trainer,
    test_dataset: torch.utils.data.dataset.Dataset,
    ignore_keys: Optional[List[str]] = None,
    metric_key_prefix: str = "test",
) -> transformers.trainer_utils.PredictionOutput:
    return trainer.predict(
        test_dataset=test_dataset,
        ignore_keys=ignore_keys,
        metric_key_prefix=metric_key_prefix,
    )


def trainer_prediction_loop(
    trainer: Trainer,
    dataloader: torch.utils.data.dataloader.DataLoader,
    description: str,
    prediction_loss_only: Optional[bool] = None,
    ignore_keys: Optional[List[str]] = None,
    metric_key_prefix: str = "eval",
) -> transformers.trainer_utils.EvalLoopOutput:
    return trainer.prediction_loop(
        dataloader=dataloader,
        description=description,
        prediction_loss_only=prediction_loss_only,
        ignore_keys=ignore_keys,
        metric_key_prefix=metric_key_prefix,
    )


def trainer_prediction_step(
    trainer: Trainer,
    model: torch.nn.modules.module.Module,
    inputs: Dict[str, Union[torch.Tensor, Any]],
    prediction_loss_only: bool,
    ignore_keys: Optional[List[str]] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    return trainer.prediction_step(
        model=model,
        inputs=inputs,
        prediction_loss_only=prediction_loss_only,
        ignore_keys=ignore_keys,
    )


def trainer_propagate_args_to_deepspeed(
    trainer: Trainer, auto_find_batch_size=False
) -> None:
    return trainer.propagate_args_to_deepspeed(
        auto_find_batch_size=auto_find_batch_size,
    )


def trainer_push_to_hub(
    trainer: Trainer,
    commit_message: Optional[str] = "End of training",
    blocking: bool = True,
    **kwargs
) -> str:
    return trainer.push_to_hub(
        commit_message=commit_message,
        blocking=blocking,
        kwargs=kwargs,
    )


def trainer_remove_callback(trainer: Trainer, callback) -> None:
    return trainer.remove_callback(
        callback=callback,
    )


def trainer_save_metrics(trainer: Trainer, split, metrics, combined=True) -> None:
    return trainer.save_metrics(
        split=split,
        metrics=metrics,
        combined=combined,
    )


def trainer_save_model(
    trainer: Trainer, output_dir: Optional[str] = None, _internal_call: bool = False
) -> None:
    return trainer.save_model(
        output_dir=output_dir,
        _internal_call=_internal_call,
    )


def trainer_save_state(trainer: Trainer) -> None:
    return trainer.save_state()


def trainer_store_flos(trainer: Trainer) -> None:
    return trainer.store_flos()


def trainer_torch_jit_model_eval(
    trainer: Trainer, model, dataloader, training=False
) -> None:
    return trainer.torch_jit_model_eval(
        model=model,
        dataloader=dataloader,
        training=training,
    )


def trainer_train(trainer: Trainer, resume_from_checkpoint: Union[bool, str, None] = None, trial: Union[ForwardRef("optuna.Trial"), Dict[str, Any]] = None, ignore_keys_for_eval: Optional[List[str]] = None, **kwargs) -> None:  # type: ignore
    return trainer.train(
        resume_from_checkpoint=resume_from_checkpoint,
        trial=trial,
        ignore_keys_for_eval=ignore_keys_for_eval,
        kwargs=kwargs,
    )


def trainer_training_step(
    trainer: Trainer,
    model: torch.nn.modules.module.Module,
    inputs: Dict[str, Union[torch.Tensor, Any]],
) -> torch.Tensor:
    return trainer.training_step(
        model=model,
        inputs=inputs,
    )
