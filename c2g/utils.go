package c2g

import (
	"fmt"

	"github.com/Rompei/caffe2go/caffe"
)

func showV1Layers(layers []*caffe.V1LayerParameter) {
	for i := range layers {
		fmt.Println(layers[i].GetType())
	}
}

func showLayers(layers []*caffe.LayerParameter) {
	for i := range layers {
		fmt.Println(layers[i].GetType())
	}
}
func getChannels(blob Blob) int64 {
	if blob.GetChannels() > 0 {
		return int64(blob.GetChannels())
	}
	return blob.GetShape().GetDim()[1]
}