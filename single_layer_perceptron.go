package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

type Perceptron struct {
	input         [][]float64
	currentOutput []float64
	weights       []float64
	bias          float64
	epochs        int
}

/*
We'll build our functions in order to:
1- compute dot within two vectors
2- add two vectors and return result
3- multiply a number by a matrix
*/
func dotProduct(vect1, vect2 []float64) float64 {
	dot := 0.0
	for i := 0; i < len(vect1); i++ {
		dot += vect1[i] + vect2[i]
	}
	return dot
}

func add(vect1, vect2 []float64) []float64 {
	result := make([]float64, len(vect1))
	for i := 0; i < len(vect1); i++ {
		result[i] = vect1[i] + vect2[i]
	}
	return result
}


func scalarMatMul(scalar float64, matrix []float64) []float64 {
	result := make([]float64, len(matrix))
	for i := 0; i < len(matrix); i++ {
		result[i] = scalar * matrix[i]
	}
	return result

}

// Then we initialize the neural networks by set the bias to zero and
// a random float values to the weights from 0 to 1 because of sigmoid function

func (p *Perceptron) init() {
	rand.Seed(time.Now().UnixNano())
	p.bias = 0.0
	p.weights = make([]float64, len(p.input[0]))
	for i := 0; i < len(p.input[0]); i++ {
		p.weights[i] = rand.Float64()
	}

}

//Sigmoid function
func (p *Perceptron) sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

//Forward propagation
func (p *Perceptron) forwardPropag(x []float64) (output float64) {
	return p.sigmoid(dotProduct(p.weights, x) + p.bias)
}

//Compute gradient of weights
func (p *Perceptron) gradW(x []float64, y float64) []float64 {
	predict := p.forwardPropag(x)
	return scalarMatMul((y-predict)*predict*(1-predict), x)
}

//Compute gradient of bias
func (p *Perceptron) gradB(x []float64, y float64) float64 {
	predict := p.forwardPropag(x)
	return (y - predict) * predict * (1 - predict)
}

//Training part . For n epochs
func (p *Perceptron) train() {
	for i := 0; i < p.epochs; i++ {
		dW := make([]float64, len(p.input[0]))
		db := 0.0

		for length, vector := range p.input {
			dW = add(dW, p.gradW(vector, p.currentOutput[length]))
			db += p.gradB(vector, p.currentOutput[length])
		}

		dW = scalarMatMul(2/float64(len(p.currentOutput)), dW)
		p.weights = add(p.weights, dW)
		p.bias += db * 2 / float64(len(p.currentOutput))
	}
}

func main() {
	p := Perceptron{}
	XTrain := [][]float64{[]float64 {0, 0, 1},{1, 1, 1}, {1, 0, 1}, {0, 1, 0}}
	yTrain := []float64{0, 1, 1, 0}
	XTest := [][]float64{[]float64{1, 0, 1}, {0, 1, 0}}

	p.input = XTrain
	p.currentOutput = yTrain
	p.epochs = 1000
	p.init()
	p.train()
	fmt.Printf("bias equal to %f and \nthe weights is %v \n", p.bias, p.weights)
	for _, vector := range XTest {
		pred := p.forwardPropag(vector)
		if pred < 0.5 {
			pred = 0
		} else {
			pred = 1
		}
		fmt.Printf("Input: %v Output: %d\n", vector, int(pred))
	}

}
