package caffe

import "github.com/lon9/caffe2go/caffe/caffeproto"

// LayerParameter is alias for LayerParameter.
type LayerParameter interface {
	GetName() string
	GetBottom() []string
	GetTop() []string
	GetLossWeight() []float32
	GetBlobs() []*caffeproto.BlobProto
	GetInclude() []*caffeproto.NetStateRule
	GetExclude() []*caffeproto.NetStateRule
	GetTransformParam() *caffeproto.TransformationParameter
	GetLossParam() *caffeproto.LossParameter
	GetAccuracyParam() *caffeproto.AccuracyParameter
	GetArgmaxParam() *caffeproto.ArgMaxParameter
	GetConcatParam() *caffeproto.ConcatParameter
	GetContrastiveLossParam() *caffeproto.ContrastiveLossParameter
	GetConvolutionParam() *caffeproto.ConvolutionParameter
	GetDataParam() *caffeproto.DataParameter
	GetDropoutParam() *caffeproto.DropoutParameter
	GetDummyDataParam() *caffeproto.DummyDataParameter
	GetEltwiseParam() *caffeproto.EltwiseParameter
	GetExpParam() *caffeproto.ExpParameter
	GetHdf5DataParam() *caffeproto.HDF5DataParameter
	GetHdf5OutputParam() *caffeproto.HDF5OutputParameter
	GetHingeLossParam() *caffeproto.HingeLossParameter
	GetImageDataParam() *caffeproto.ImageDataParameter
	GetInfogainLossParam() *caffeproto.InfogainLossParameter
	GetInnerProductParam() *caffeproto.InnerProductParameter
	GetLrnParam() *caffeproto.LRNParameter
	GetMemoryDataParam() *caffeproto.MemoryDataParameter
	GetMvnParam() *caffeproto.MVNParameter
	GetPoolingParam() *caffeproto.PoolingParameter
	GetPowerParam() *caffeproto.PowerParameter
	GetReluParam() *caffeproto.ReLUParameter
	GetSigmoidParam() *caffeproto.SigmoidParameter
	GetSoftmaxParam() *caffeproto.SoftmaxParameter
	GetSliceParam() *caffeproto.SliceParameter
	GetTanhParam() *caffeproto.TanHParameter
	GetThresholdParam() *caffeproto.ThresholdParameter
	GetWindowDataParam() *caffeproto.WindowDataParameter
}

// Parameter is alias of parameter.
type Parameter interface {
	GetKernelH() uint32
	GetKernelW() uint32
	GetKernelSize() uint32
	GetStrideH() uint32
	GetStrideW() uint32
	GetStride() uint32
	GetPad() uint32
	GetPadH() uint32
	GetPadW() uint32
}

// Blob is alias of Blob.
type Blob interface {
	GetNum() int32
	GetShape() *caffeproto.BlobShape
	GetHeight() int32
	GetWidth() int32
	GetChannels() int32
}
