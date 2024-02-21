from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # `TYPE_CHECKING` is for code hints, if you don't need, just delete this part
    from .configuration_deberta import DeBERTaConfig
    from .modeling_deberta import DebertaModel
    from .tokenization_deberta import DeBERTaTokenizer
