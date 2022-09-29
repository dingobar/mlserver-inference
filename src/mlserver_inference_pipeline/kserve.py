from enum import Enum, auto


class KserveDataType(str, Enum):
    # adapted from https://kserve.github.io/website/modelserving/inference_api/#tensor-data-types
    BOOL = "bool"
    UINT8 = "uint8"
    UINT16 = "uint16"
    UINT32 = "uint32"
    UINT64 = "uint64"
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"
    FP16 = "fp16"
    FP32 = "fp32"
    FP64 = "fp64"
    BYTES = "bytes"
