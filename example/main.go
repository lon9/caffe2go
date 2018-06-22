package main

import (
	"fmt"
	_ "image/jpeg"
	_ "image/png"

	"github.com/lon9/caffe2go/caffe"
)

func main() {
	caffe2go, err := caffe.NewCaffe2Go("lenet.caffemodel")
	if err != nil {
		panic(err)
	}
	output, err := caffe2go.Predict("mnist_zero.png", 28, nil)
	if err != nil {
		panic(err)
	}

	for i := range output {
		fmt.Printf("%d: %f\n", i, output[i][0][0])
	}

}
