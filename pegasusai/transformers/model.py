from enum import Enum

from transformers import (
    AutoModel,
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
)


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


def load_checkpoint(model_id: str, task: Task, **kwargs) -> AutoModel:
    return task.value.from_pretrained(model_id, **kwargs)
