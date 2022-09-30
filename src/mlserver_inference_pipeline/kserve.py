from enum import Enum, auto


class KserveDataType(str, Enum):
    # adapted from https://kserve.github.io/website/modelserving/inference_api/#tensor-data-types
    BOOL = "BOOL"
    UINT8 = "UINT8"
    UINT16 = "UINT16"
    UINT32 = "UINT32"
    UINT64 = "UINT64"
    INT8 = "INT8"
    INT16 = "INT16"
    INT32 = "INT32"
    INT64 = "INT64"
    FP16 = "FP16"
    FP32 = "FP32"
    FP64 = "FP64"
    BYTES = "BYTES"
