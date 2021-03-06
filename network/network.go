package network

import (
	"fmt"

	"github.com/lon9/caffe2go/layers"
)

// Network have netword definition.
type Network struct {
	layers []layers.Layer
}

// Add adds layer to network.
func (n *Network) Add(layer layers.Layer) {
	n.layers = append(n.layers, layer)
}

// Predict forwards network.
func (n *Network) Predict(input [][][]float32) (output [][][]float32, err error) {
	for i := range n.layers {
		fmt.Println(n.layers[i].GetName())
		fmt.Println(len(input), len(input[0]))
		input, err = n.layers[i].Forward(input)
		if err != nil {
			return
		}
	}
	output = input
	return
}
