from enum import Enum


MOSI_MODALITIES = {
    "visual": "CMU_MOSI_Visual_Facet_42",
    # "visual": "CMU_MOSI_Visual_OpenFace_1",
    # "audio": "CMU_MOSI_openSMILE_IS09",
    "audio": "CMU_MOSI_COVAREP",
    "text": "CMU_MOSI_TimestampedWords",
    "glove": "CMU_MOSI_TimestampedWordVectors",
    "labels": "CMU_MOSI_Opinion_Labels",
}

MOSEI_MODALITIES = {
    "text": "CMU_MOSEI_TimestampedWords",
    "audio": "CMU_MOSEI_COVAREP",
    # "audio": "CMU_MOSEI_openSMILE_IS09",
    "phones": "CMU_MOSEI_TimestampedPhones",
    "visual": "CMU_MOSEI_VisualFacet42",
    "glove": "CMU_MOSEI_TimestampedWordVectors",
    "labels": "CMU_MOSEI_Labels",
}

MOSEI_MODALITIES2 = {
    "text": "CMU_MOSEI_TimestampedWords",
    "audio": "CMU_MOSEI_COVAREP",
    # "audio": "CMU_MOSEI_openSMILE_IS09",
    "phones": "CMU_MOSEI_TimestampedPhones",
    "visual": "CMU_MOSEI_VisualOpenFace2",
    "glove": "CMU_MOSEI_TimestampedWordVectors",
    "labels": "CMU_MOSEI_Labels",
}


class SPECIAL_TOKENS(Enum):
    PAD = "[PAD]"
    MASK = "[MASK]"
    UNK = "[UNK]"
    BOS = "[BOS]"
    EOS = "[EOS]"
    CLS = "[CLS]"
    PAUSE = "[PAUSE]"

    @classmethod
    def has_token(cls, token):
        return any(token == t.name or token == t.value for t in cls)

    @classmethod
    def to_list(cls):
        return list(map(lambda x: x.value, cls))
