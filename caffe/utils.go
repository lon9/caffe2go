package caffe

import (
	"fmt"

	"github.com/lon9/caffe2go/caffe/caffeproto"
)

func showV1Layers(layers []*caffeproto.V1LayerParameter) {
	for i := range layers {
		fmt.Println(layers[i].GetType())
	}
}

func showLayers(layers []*caffeproto.LayerParameter) {
	for i := range layers {
		fmt.Println(layers[i].GetType())
	}
}
