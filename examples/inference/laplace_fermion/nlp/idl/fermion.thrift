include "base/base.thrift"
namespace cpp idl.fermion.core
namespace go idl.fermion.core

enum ErrorCode {
  OK = 0,

  // 6661000 - 6661999: invalid request, don't retry
  //   6661000: general error
  //   6661100 - 6661499: transformer error
  //   6661500 - 6661799: predictor error
  //   6661800: tensor error
  //   6661900: handler error
  INVALID_REQUEST = 6661000,

  // 6662000 - 6662999: server error
  //   6662000: general error
  //   6662100 - 6662499: transformer error
  //   6662500 - 6662799: predictor error
  //   6662800: tensor error
  //   6662900: handler error
  SERVER_ERROR = 6662000,

  // Unknown exception happen
  UNKNOWN = 6669999,
}

enum ModalType {
  UNKNOWN = 0,
  IMAGE_DATA = 1,
  IMAGE_URL = 2,
  TEXT = 3,
  FEATURE = 4,
  AUDIO_DATA = 5,
  AUDIO_URL = 6,
}

struct TensorAttr {
  1: list<i32> shape,
  2: DataType data_type,
  3: ModalType modal_type,
  4: string description,
  5: bool is_optional,
}

enum DataType {
  // Reference: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/types.proto
  // Not a legal value for DataType.  Used to indicate a DataType field
  // has not been set.
  INVALID = 0,

  // Common data types.
  INT8 = 8, // use int8_data
  INT16 = 16, // use int16_data
  INT32 = 32, // use int32_data
  INT64 = 64, // use int64_data
  UINT8 = 108,
  UINT16 = 116,
  UINT32 = 132,
  UINT64 = 164,
  FLOAT16 = 216,
  FLOAT32 = 232, // use float_data
  FLOAT64 = 264, // use float_data

  // Byte strings.
  STRING = 300, // use str_data

  DATA_INT8 = 1008, // use data to store int8
  DATA_INT16 = 1016, // use data to store int16
  DATA_INT32 = 1032, // use data to store int32
  DATA_INT64 = 1064, // use data to store int64
  DATA_UINT8 = 1108, // use data to store uint8
  DATA_UINT16 = 1116, // use data to store uint16
  DATA_UINT32 = 1132, // use data to store uint32
  DATA_UINT64 = 1164, // use data to store uint64
  DATA_FLOAT16 = 1216, // use data to store float16
  DATA_FLOAT32 = 1232, // use data to store float32
  DATA_FLOAT64 = 1264, // use data to store float64
}

struct Tensor {
  // len(xx_data) = shape[0] * shape[1] * ... * shape[-1]
  1: DataType dtype,
  2: list<i32> shape,

  // data fields.
  3: list<byte> int8_data,
  4: list<i16> int16_data,
  5: list<i32> int32_data,
  6: list<i64> int64_data,
  7: list<double> float_data,
  8: list<binary> str_data,
  9: binary data,


  // Extra metadata for tensor, like vid, image url.
  255: map<string, string> extra,
}

// A set of named tensors, associated with a single input item.
struct TensorSet {
  1: map<string, Tensor> tensors, //

  // Extra metadata for tensor, like vid, image url.
  255: map<string, string> extra,
}

struct InferRequest {
  1: list<TensorSet> input, // A batch of input item
  2: string model_name,
  3: optional bool disable_cache,
  254: map<string, string> extra,
  255: optional base.Base Base,
}

struct InferResponse {
  1: list<TensorSet> output, // A batch of output item
  2: string model_version,
  254: map<string, string> extra,
  255: optional base.BaseResp BaseResp,
}

struct ModelInfoRequest {
  1: string model_name,
  255: optional base.Base Base,
}

struct ModelInfoResponse {
  1: string model_version,
  2: string model_type,
  3: string model_description,
  4: map<string, TensorAttr> input,
  5: map<string, TensorAttr> output,
  255: optional base.BaseResp BaseResp,
}

struct ModelListRequest {
  255: optional base.Base Base,
}

struct ModelListResponse {
  1: map<string, ModelInfoResponse> model_info,
  255: optional base.BaseResp BaseResp,
}
service FermionCore {
  InferResponse Infer(1: InferRequest request),
  ModelInfoResponse ModelInfo(1: ModelInfoRequest request),
  ModelListResponse ModelList(1: ModelListRequest request),
}
