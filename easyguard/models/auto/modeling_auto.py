# Very heavily inspired by:
# https://code.byted.org/lab/titan/blob/master/titan/models/auto/modeling_auto.py

from cruise.utilities.logger import get_cruise_logger
from transformers import AutoBackbone as tsfm_AutoBackbone
from transformers import AutoModel as tsfm_AutoModel
from transformers import AutoModelForAudioClassification as tsfm_AutoModelForAudioClassification
from transformers import AutoModelForAudioFrameClassification as tsfm_AutoModelForAudioFrameClassification
from transformers import AutoModelForAudioXVector as tsfm_AutoModelForAudioXVector
from transformers import AutoModelForCausalLM as tsfm_AutoModelForCausalLM
from transformers import AutoModelForCTC as tsfm_AutoModelForCTC
from transformers import AutoModelForDepthEstimation as tsfm_AutoModelForDepthEstimation
from transformers import AutoModelForDocumentQuestionAnswering as tsfm_AutoModelForDocumentQuestionAnswering
from transformers import AutoModelForImageClassification as tsfm_AutoModelForImageClassification
from transformers import AutoModelForImageSegmentation as tsfm_AutoModelForImageSegmentation
from transformers import AutoModelForInstanceSegmentation as tsfm_AutoModelForInstanceSegmentation
from transformers import AutoModelForMaskedImageModeling as tsfm_AutoModelForMaskedImageModeling
from transformers import AutoModelForMaskedLM as tsfm_AutoModelForMaskedLM
from transformers import AutoModelForMultipleChoice as tsfm_AutoModelForMultipleChoice
from transformers import AutoModelForNextSentencePrediction as tsfm_AutoModelForNextSentencePrediction
from transformers import AutoModelForObjectDetection as tsfm_AutoModelForObjectDetection
from transformers import AutoModelForPreTraining as tsfm_AutoModelForPreTraining
from transformers import AutoModelForQuestionAnswering as tsfm_AutoModelForQuestionAnswering
from transformers import AutoModelForSemanticSegmentation as tsfm_AutoModelForSemanticSegmentation
from transformers import AutoModelForSeq2SeqLM as tsfm_AutoModelForSeq2SeqLM
from transformers import AutoModelForSequenceClassification as tsfm_AutoModelForSequenceClassification
from transformers import AutoModelForSpeechSeq2Seq as tsfm_AutoModelForSpeechSeq2Seq
from transformers import AutoModelForTableQuestionAnswering as tsfm_AutoModelForTableQuestionAnswering
from transformers import AutoModelForTokenClassification as tsfm_AutoModelForTokenClassification
from transformers import AutoModelForUniversalSegmentation as tsfm_AutoModelForUniversalSegmentation
from transformers import AutoModelForVideoClassification as tsfm_AutoModelForVideoClassification
from transformers import AutoModelForVision2Seq as tsfm_AutoModelForVision2Seq
from transformers import AutoModelForVisualQuestionAnswering as tsfm_AutoModelForVisualQuestionAnswering
from transformers import AutoModelForZeroShotImageClassification as tsfm_AutoModelForZeroShotImageClassification
from transformers import AutoModelForZeroShotObjectDetection as tsfm_AutoModelForZeroShotObjectDetection

logger = get_cruise_logger()


class EasyGuardFromPretrained:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        pass


class AutoModel(EasyGuardFromPretrained, tsfm_AutoModel):
    logger.debug("transformers AutoModel is replaced with EasyGuard AutoModel")


class AutoModelForPreTraining(EasyGuardFromPretrained, tsfm_AutoModelForPreTraining):
    logger.debug("transformers AutoModelForPreTraining is replaced with EasyGuard AutoModelForPreTraining")


class AutoModelForCausalLM(EasyGuardFromPretrained, tsfm_AutoModelForCausalLM):
    logger.debug("transformers AutoModelForCausalLM is replaced with EasyGuard AutoModelForCausalLM")


class AutoModelForMaskedLM(EasyGuardFromPretrained, tsfm_AutoModelForMaskedLM):
    logger.debug("transformers AutoModelForMaskedLM is replaced with EasyGuard AutoModelForMaskedLM")


class AutoModelForSeq2SeqLM(EasyGuardFromPretrained, tsfm_AutoModelForSeq2SeqLM):
    logger.debug("transformers AutoModelForSeq2SeqLM is replaced with EasyGuard AutoModelForSeq2SeqLM")


class AutoModelForSequenceClassification(EasyGuardFromPretrained, tsfm_AutoModelForSequenceClassification):
    logger.debug(
        "transformers AutoModelForSequenceClassification is replaced with EasyGuard AutoModelForSequenceClassification"
    )


class AutoModelForQuestionAnswering(EasyGuardFromPretrained, tsfm_AutoModelForQuestionAnswering):
    logger.debug("transformers AutoModelForQuestionAnswering is replaced with EasyGuard AutoModelForQuestionAnswering")


class AutoModelForTableQuestionAnswering(EasyGuardFromPretrained, tsfm_AutoModelForTableQuestionAnswering):
    logger.debug(
        "transformers AutoModelForTableQuestionAnswering is replaced with EasyGuard AutoModelForTableQuestionAnswering"
    )


class AutoModelForVisualQuestionAnswering(EasyGuardFromPretrained, tsfm_AutoModelForVisualQuestionAnswering):
    logger.debug(
        "transformers AutoModelForVisualQuestionAnswering is replaced with "
        "EasyGuard AutoModelForVisualQuestionAnswering"
    )


class AutoModelForDocumentQuestionAnswering(EasyGuardFromPretrained, tsfm_AutoModelForDocumentQuestionAnswering):
    logger.debug(
        "transformers AutoModelForDocumentQuestionAnswering is replaced with "
        "EasyGuard AutoModelForDocumentQuestionAnswering"
    )


class AutoModelForTokenClassification(EasyGuardFromPretrained, tsfm_AutoModelForTokenClassification):
    logger.debug(
        "transformers AutoModelForTokenClassification is replaced with EasyGuard AutoModelForTokenClassification"
    )


class AutoModelForMultipleChoice(EasyGuardFromPretrained, tsfm_AutoModelForMultipleChoice):
    logger.debug("transformers AutoModelForMultipleChoice is replaced with EasyGuard AutoModelForMultipleChoice")


class AutoModelForNextSentencePrediction(EasyGuardFromPretrained, tsfm_AutoModelForNextSentencePrediction):
    logger.debug(
        "transformers AutoModelForNextSentencePrediction is replaced with EasyGuard AutoModelForNextSentencePrediction"
    )


class AutoModelForImageClassification(EasyGuardFromPretrained, tsfm_AutoModelForImageClassification):
    logger.debug(
        "transformers AutoModelForImageClassification is replaced with EasyGuard AutoModelForImageClassification"
    )


class AutoModelForZeroShotImageClassification(EasyGuardFromPretrained, tsfm_AutoModelForZeroShotImageClassification):
    logger.debug(
        "transformers AutoModelForZeroShotImageClassification is replaced with "
        "EasyGuard AutoModelForZeroShotImageClassification"
    )


class AutoModelForImageSegmentation(EasyGuardFromPretrained, tsfm_AutoModelForImageSegmentation):
    logger.debug("transformers AutoModelForImageSegmentation is replaced with EasyGuard AutoModelForImageSegmentation")


class AutoModelForSemanticSegmentation(EasyGuardFromPretrained, tsfm_AutoModelForSemanticSegmentation):
    logger.debug(
        "transformers AutoModelForSemanticSegmentation is replaced with EasyGuard AutoModelForSemanticSegmentation"
    )


class AutoModelForUniversalSegmentation(EasyGuardFromPretrained, tsfm_AutoModelForUniversalSegmentation):
    logger.debug(
        "transformers AutoModelForUniversalSegmentation is replaced with EasyGuard AutoModelForUniversalSegmentation"
    )


class AutoModelForInstanceSegmentation(EasyGuardFromPretrained, tsfm_AutoModelForInstanceSegmentation):
    logger.debug(
        "transformers AutoModelForInstanceSegmentation is replaced with EasyGuard AutoModelForInstanceSegmentation"
    )


class AutoModelForObjectDetection(EasyGuardFromPretrained, tsfm_AutoModelForObjectDetection):
    logger.debug("transformers AutoModelForObjectDetection is replaced with EasyGuard AutoModelForObjectDetection")


class AutoModelForZeroShotObjectDetection(EasyGuardFromPretrained, tsfm_AutoModelForZeroShotObjectDetection):
    logger.debug(
        "transformers AutoModelForZeroShotObjectDetection is replaced with "
        "EasyGuard AutoModelForZeroShotObjectDetection"
    )


class AutoModelForDepthEstimation(EasyGuardFromPretrained, tsfm_AutoModelForDepthEstimation):
    logger.debug("transformers AutoModelForDepthEstimation is replaced with EasyGuard AutoModelForDepthEstimation")


class AutoModelForVideoClassification(EasyGuardFromPretrained, tsfm_AutoModelForVideoClassification):
    logger.debug(
        "transformers AutoModelForVideoClassification is replaced with EasyGuard AutoModelForVideoClassification"
    )


class AutoModelForVision2Seq(EasyGuardFromPretrained, tsfm_AutoModelForVision2Seq):
    logger.debug("transformers AutoModelForVision2Seq is replaced with EasyGuard AutoModelForVision2Seq")


class AutoModelForAudioClassification(EasyGuardFromPretrained, tsfm_AutoModelForAudioClassification):
    logger.debug(
        "transformers AutoModelForAudioClassification is replaced with EasyGuard AutoModelForAudioClassification"
    )


class AutoModelForCTC(EasyGuardFromPretrained, tsfm_AutoModelForCTC):
    logger.debug("transformers AutoModelForCTC is replaced with EasyGuard AutoModelForCTC")


class AutoModelForSpeechSeq2Seq(EasyGuardFromPretrained, tsfm_AutoModelForSpeechSeq2Seq):
    logger.debug("transformers AutoModelForSpeechSeq2Seq is replaced with EasyGuard AutoModelForSpeechSeq2Seq")


class AutoModelForAudioFrameClassification(EasyGuardFromPretrained, tsfm_AutoModelForAudioFrameClassification):
    logger.debug(
        "transformers AutoModelForAudioFrameClassification is replaced with "
        "EasyGuard AutoModelForAudioFrameClassification"
    )


class AutoModelForAudioXVector(EasyGuardFromPretrained, tsfm_AutoModelForAudioXVector):
    logger.debug("transformers AutoModelForAudioXVector is replaced with EasyGuard AutoModelForAudioXVector")


class AutoBackbone(EasyGuardFromPretrained, tsfm_AutoBackbone):
    logger.debug("transformers AutoBackbone is replaced with EasyGuard AutoBackbone")


class AutoModelForMaskedImageModeling(EasyGuardFromPretrained, tsfm_AutoModelForMaskedImageModeling):
    logger.debug(
        "transformers AutoModelForMaskedImageModeling is replaced with EasyGuard AutoModelForMaskedImageModeling"
    )
