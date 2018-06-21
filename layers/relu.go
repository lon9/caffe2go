package layers

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// ReLULayer is layer of ReLU.
type ReLULayer struct {
	*BaseLayer
	Slope float32
}

// NewReLULayer is constructor.
func NewReLULayer(name, t string, slope float32) *ReLULayer {
	return &ReLULayer{
		BaseLayer: NewBaseLayer(name, t),
		Slope:     slope,
	}
}

// Forward forwards a step.
func (r *ReLULayer) Forward(input [][][]float32) ([][][]float32, error) {
	doneCh := make(chan bool, len(input))
	for i := range input {
		go func(idx int) {
			t := ConvertMatrix(input[idx])
			var out mat.Dense
			out.Apply(func(i, j int, v float64) float64 {
				return math.Max(0, v)
			}, t)
			input[idx] = ConvertMat64(&out)
			doneCh <- true
		}(i)
	}
	for range input {
		<-doneCh
	}
	close(doneCh)
	return input, nil
}
