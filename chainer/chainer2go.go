package chainer

import (
	"fmt"
	"strings"

	"github.com/lon9/caffe2go/layers"
	"github.com/lon9/caffe2go/network"
	"gonum.org/v1/hdf5"
)

// Chainer2Go is interface of chainer2go.
type Chainer2Go struct {
	Network *network.Network
}

// NewChainer2Go is constructor.
func NewChainer2Go(modelPath string) (c2g *Chainer2Go, err error) {
	f, err := hdf5.OpenFile(modelPath, hdf5.F_ACC_RDONLY)
	if err != nil {
		return
	}
	predictor, err := f.OpenGroup("predictor")
	if err != nil {
		return
	}

	num, err := predictor.NumObjects()
	if err != nil {
		return
	}

	var net network.Network
	for i := 0; i < int(num); i++ {
		objName, err := predictor.ObjectNameByIndex(uint(i))
		if err != nil {
			return nil, err
		}

		if strings.HasPrefix(objName, "conv") {
			fmt.Println(layers.Convolution)
			convLayer, err := SetupConvolution(objName, predictor)
			if err != nil {
				return nil, err
			}
			net.Add(convLayer)
			fmt.Println()
			continue
		}

		if strings.HasPrefix(objName, "fc") {
			continue
		}
	}

	// Close predictor
	if err = predictor.Close(); err != nil {
		return
	}

	return &Chainer2Go{
		Network: &net,
	}, nil
}
