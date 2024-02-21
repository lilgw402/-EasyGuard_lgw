from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # `TYPE_CHECKING` is for code hints, if you don't need, just delete this part
    from .configuration_deberta_v2 import DebertaV2Config
    from .modeling_deberta_v2 import DebertaV2Model
    from .tokenization_deberta_v2 import DebertaV2Tokenizer
