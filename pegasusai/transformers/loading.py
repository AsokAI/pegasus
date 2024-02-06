from enum import Enum

from transformers import (
    AutoModel,
    AutoBackbone,
    AutoConfig,
    AutoTokenizer,
    AutoFeatureExtractor,
    AutoImageProcessor,
    AutoProcessor,
    AutoModelForAudioClassification,
    AutoModelForAudioFrameClassification,
    AutoModelForAudioXVector,
    AutoModelForCTC,
    AutoModelForCausalLM,
    AutoModelForDepthEstimation,
    AutoModelForDocumentQuestionAnswering,
    AutoModelForImageClassification,
    AutoModelForImageSegmentation,
    AutoModelForImageToImage,
    AutoModelForInstanceSegmentation,
    AutoModelForMaskGeneration,
    AutoModelForMaskedImageModeling,
    AutoModelForMaskedLM,
    AutoModelForMultipleChoice,
    AutoModelForNextSentencePrediction,
    AutoModelForObjectDetection,
    AutoModelForPreTraining,
    AutoModelForQuestionAnswering,
    AutoModelForSemanticSegmentation,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForSpeechSeq2Seq,
    AutoModelForTableQuestionAnswering,
    AutoModelForTextEncoding,
    AutoModelForTextToSpectrogram,
    AutoModelForTextToWaveform,
    AutoModelForTokenClassification,
    AutoModelForUniversalSegmentation,
    AutoModelForVideoClassification,
    AutoModelForVision2Seq,
    AutoModelForVisualQuestionAnswering,
    AutoModelForZeroShotImageClassification,
    AutoModelForZeroShotObjectDetection,
    GenerationConfig,
    pipeline,
)
from transformers.pipelines.base import Pipeline


class Task(Enum):
    AudioClassification = AutoModelForAudioClassification
    AudioFrameClassification = AutoModelForAudioFrameClassification
    AudioXVector = AutoModelForAudioXVector
    CTC = AutoModelForCTC
    CausalLM = AutoModelForCausalLM
    DepthEstimation = AutoModelForDepthEstimation
    DocumentQuestionAnswering = AutoModelForDocumentQuestionAnswering
    ImageClassification = AutoModelForImageClassification
    ImageSegmentation = AutoModelForImageSegmentation
    ImageToImage = AutoModelForImageToImage
    InstanceSegmentation = AutoModelForInstanceSegmentation
    MaskGeneration = AutoModelForMaskGeneration
    MaskedImageModeling = AutoModelForMaskedImageModeling
    MaskedLM = AutoModelForMaskedLM
    MultipleChoice = AutoModelForMultipleChoice
    NextSentencePrediction = AutoModelForNextSentencePrediction
    ObjectDetection = AutoModelForObjectDetection
    PreTraining = AutoModelForPreTraining
    QuestionAnswering = AutoModelForQuestionAnswering
    SemanticSegmentation = AutoModelForSemanticSegmentation
    Seq2SeqLM = AutoModelForSeq2SeqLM
    SequenceClassification = AutoModelForSequenceClassification
    SpeechSeq2Seq = AutoModelForSpeechSeq2Seq
    TableQuestionAnswering = AutoModelForTableQuestionAnswering
    TextEncoding = AutoModelForTextEncoding
    TextToSpectrogram = AutoModelForTextToSpectrogram
    TextToWaveform = AutoModelForTextToWaveform
    TokenClassification = AutoModelForTokenClassification
    UniversalSegmentation = AutoModelForUniversalSegmentation
    VideoClassification = AutoModelForVideoClassification
    Vision2Seq = AutoModelForVision2Seq
    VisualQuestionAnswering = AutoModelForVisualQuestionAnswering
    ZeroShotImageClassification = AutoModelForZeroShotImageClassification
    ZeroShotObjectDetection = AutoModelForZeroShotObjectDetection


def load_model(repo_id: str, task: Task, **kwargs) -> AutoModel:
    return task.value.from_pretrained(repo_id, **kwargs)


def load_backbone(repo_id: str, **kwargs) -> AutoBackbone:
    return AutoBackbone.from_pretrained(repo_id, **kwargs)


def load_config(repo_id: str, **kwargs) -> AutoConfig:
    return AutoConfig.from_pretrained(repo_id, **kwargs)


def load_tokenizer(repo_id: str, **kwargs) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(repo_id, **kwargs)


def load_auto_feature_extractor(repo_id: str, **kwargs) -> AutoFeatureExtractor:
    return AutoFeatureExtractor.from_pretrained(repo_id, **kwargs)


def load_auto_image_processor(repo_id: str, **kwargs) -> AutoImageProcessor:
    return AutoImageProcessor.from_pretrained(repo_id, **kwargs)


def load_processor(repo_id: str, **kwargs) -> AutoProcessor:
    return AutoProcessor.from_pretrained(repo_id, **kwargs)


def load_generation_config(repo_id: str, **kwargs) -> GenerationConfig:
    return GenerationConfig.from_pretrained(repo_id, **kwargs)


def load_pipeline(repo_id: str, **kwargs) -> Pipeline:
    return pipeline(repo_id, **kwargs)
