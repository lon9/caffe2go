package chainer

import (
	"errors"

	"github.com/lon9/caffe2go/layers"
	"gonum.org/v1/hdf5"
)

// SetupConvolution setups ConvolutionLayer from Chainer model.
func SetupConvolution(name string, predictor *hdf5.Group) (*layers.ConvolutionLayer, error) {

	layer, err := predictor.OpenGroup(name)
	if err != nil {
		return nil, err
	}

	num, err := layer.NumObjects()
	if err != nil {
		return nil, err
	} else if num != 2 {
		return nil, errors.New("Layer dataset size must be 2")
	}

	var (
		nIn        uint32
		nOut       uint32
		kernelSize int
		stride     int
		padding    int
	)

	for i := 0; i < int(num); i++ {
		objName, err := layer.ObjectNameByIndex(0)
		if err != nil {
			return nil, err
		}

		switch objName {
		case "W":
			dset, err := layer.OpenDataset(objName)
			if err != nil {
				return nil, err
			}
			space := dset.Space()
			dims, _, err := space.SimpleExtentDims()
			if err != nil {
				return nil, err
			}
			if len(dims) != 4 {
				return nil, errors.New("W must be 4 dimensions")
			}
			nOut = uint32(dims[0])
			nIn = uint32(dims[1])
			kernelSize = int(dims[2])
		case "b":
			dset, err := layer.OpenDataset(objName)
			if err != nil {
				return nil, err
			}
		}
	}
	if err := layer.Close(); err != nil {
		return nil, err
	}
}
