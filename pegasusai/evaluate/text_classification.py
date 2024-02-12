import numbers
from typing import (
    Any,
    Callable,
    Dict,
    ForwardRef,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import datasets
import evaluate
from evaluate import TextClassificationEvaluator


def text_classification_evaluator___init__(
    text_classification_evaluator: TextClassificationEvaluator,
    task="text-classification",
    default_metric_name=None,
) -> None:
    return text_classification_evaluator.__init__(
        task=task,
        default_metric_name=default_metric_name,
    )


def text_classification_evaluator__compute_confidence_interval(
    text_classification_evaluator: TextClassificationEvaluator,
    metric,
    metric_inputs,
    metric_keys: List[str],
    confidence_level: float = 0.95,
    n_resamples: int = 9999,
    random_state: Optional[int] = None,
):
    return text_classification_evaluator._compute_confidence_interval(
        metric=metric,
        metric_inputs=metric_inputs,
        metric_keys=metric_keys,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        random_state=random_state,
    )


def text_classification_evaluator__compute_time_perf(
    text_classification_evaluator: TextClassificationEvaluator,
    start_time: float,
    end_time: float,
    num_samples: int,
):
    return text_classification_evaluator._compute_time_perf(
        start_time=start_time,
        end_time=end_time,
        num_samples=num_samples,
    )


def text_classification_evaluator__infer_device(
    text_classification_evaluator: TextClassificationEvaluator,
):
    return text_classification_evaluator._infer_device()


def text_classification_evaluator_call_pipeline(
    text_classification_evaluator: TextClassificationEvaluator, pipe, *args, **kwargs
) -> None:
    return text_classification_evaluator.call_pipeline(
        pipe=pipe,
        args=args,
        kwargs=kwargs,
    )


def text_classification_evaluator_check_for_mismatch_in_device_setup(
    text_classification_evaluator: TextClassificationEvaluator,
    device,
    model_or_pipeline,
) -> None:
    return text_classification_evaluator.check_for_mismatch_in_device_setup(
        device=device,
        model_or_pipeline=model_or_pipeline,
    )


def text_classification_evaluator_check_required_columns(
    text_classification_evaluator: TextClassificationEvaluator,
    data: Union[str, datasets.arrow_dataset.Dataset],
    columns_names: Dict[str, str],
) -> None:
    return text_classification_evaluator.check_required_columns(
        data=data,
        columns_names=columns_names,
    )


def text_classification_evaluator_compute(text_classification_evaluator: TextClassificationEvaluator, model_or_pipeline: Union[str, ForwardRef("Pipeline"), Callable, ForwardRef("PreTrainedModel"), ForwardRef("TFPreTrainedModel")] = None, data: Union[str, datasets.arrow_dataset.Dataset] = None, subset: Optional[str] = None, split: Optional[str] = None, metric: Union[str, evaluate.module.EvaluationModule] = None, tokenizer: Union[str, ForwardRef("PreTrainedTokenizer"), None] = None, feature_extractor: Union[str, ForwardRef("FeatureExtractionMixin"), None] = None, strategy: Literal["simple", "bootstrap"] = "simple", confidence_level: float = 0.95, n_resamples: int = 9999, device: int = None, random_state: Optional[int] = None, input_column: str = "text", second_input_column: Optional[str] = None, label_column: str = "label", label_mapping: Optional[Dict[str, numbers.Number]] = None) -> Tuple[Dict[str, float], Any]:  # type: ignore
    return text_classification_evaluator.compute(
        model_or_pipeline=model_or_pipeline,
        data=data,
        subset=subset,
        split=split,
        metric=metric,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        strategy=strategy,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        device=device,
        random_state=random_state,
        input_column=input_column,
        second_input_column=second_input_column,
        label_column=label_column,
        label_mapping=label_mapping,
    )


def text_classification_evaluator_compute_metric(
    text_classification_evaluator: TextClassificationEvaluator,
    metric: evaluate.module.EvaluationModule,
    metric_inputs: Dict,
    strategy: Literal["simple", "bootstrap"] = "simple",
    confidence_level: float = 0.95,
    n_resamples: int = 9999,
    random_state: Optional[int] = None,
) -> None:
    return text_classification_evaluator.compute_metric(
        metric=metric,
        metric_inputs=metric_inputs,
        strategy=strategy,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        random_state=random_state,
    )


def text_classification_evaluator_get_dataset_split(
    text_classification_evaluator: TextClassificationEvaluator,
    data,
    subset,
    split,
) -> None:
    return text_classification_evaluator.get_dataset_split(
        data=data,
        subset=subset,
        split=split,
    )


def text_classification_evaluator_load_data(
    text_classification_evaluator: TextClassificationEvaluator,
    data: Union[str, datasets.arrow_dataset.Dataset],
    subset: str = None,
    split: str = None,
) -> None:
    return text_classification_evaluator.load_data(
        data=data,
        subset=subset,
        split=split,
    )


def text_classification_evaluator_predictions_processor(
    text_classification_evaluator: TextClassificationEvaluator,
    predictions,
    label_mapping,
) -> None:
    return text_classification_evaluator.predictions_processor(
        predictions=predictions,
        label_mapping=label_mapping,
    )


def text_classification_evaluator_prepare_data(
    text_classification_evaluator: TextClassificationEvaluator,
    data: Union[str, datasets.arrow_dataset.Dataset],
    input_column: str,
    second_input_column: str,
    label_column: str,
) -> None:
    return text_classification_evaluator.prepare_data(
        data=data,
        input_column=input_column,
        second_input_column=second_input_column,
        label_column=label_column,
    )


def text_classification_evaluator_prepare_metric(
    text_classification_evaluator: TextClassificationEvaluator,
    metric: Union[str, evaluate.module.EvaluationModule],
) -> None:
    return text_classification_evaluator.prepare_metric(
        metric=metric,
    )


def text_classification_evaluator_prepare_pipeline(text_classification_evaluator: TextClassificationEvaluator, model_or_pipeline: Union[str, ForwardRef("Pipeline"), Callable, ForwardRef("PreTrainedModel"), ForwardRef("TFPreTrainedModel")], tokenizer: Union[ForwardRef("PreTrainedTokenizerBase"), ForwardRef("FeatureExtractionMixin")] = None, feature_extractor: Union[ForwardRef("PreTrainedTokenizerBase"), ForwardRef("FeatureExtractionMixin")] = None, device: int = None) -> None:  # type: ignore
    return text_classification_evaluator.prepare_pipeline(
        model_or_pipeline=model_or_pipeline,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        device=device,
    )
