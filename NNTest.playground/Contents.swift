import Foundation
import Accelerate


//let mnist = Bundle.main.decode(Dataset.self, from: "mnistSmall.json")



let network = NeuralNetwork(layers: [
	DenseLayer(inputSize: 2, outputSize: 3, activation: Swish()),
	DenseLayer(inputSize: 3, outputSize: 1, activation: Swish())
])


let x = Matrix([[0, 0], [1, 1], [1, 0], [0, 1]])
let y = Matrix([[0], [1], [0], [0]])

//let x = Matrix([
//	[0, 0, 1],
//	[0, 1, 1],
//	[1, 0, 0],
//	[1, 1, 0],
//	[1, 0, 1],
//	[1, 1, 1],
//])
//
//let y = Matrix([
//	[0],
//	[1],
//	[0],
//	[1],
//	[1],
//	[0]
//])


//let x : [Matrix] = mnist.x_train.chunked(into: chunkSize).compactMap { Matrix($0) }
//let y : [Matrix] = mnist.y_train.chunked(into: chunkSize).compactMap { Matrix($0) }
let chunkSize = 64
network.fit(
	x: x,
	y: y,
	epochs: 10000,
	learningRate: 0.1,
	batchSize: 64
)


//[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
print(network.predict(x: x).formattedData())



//
//let nNeurons = 2
//var w1 = Matrix.random(
//	size: MatrixSize(rows: x.size().columns, columns : nNeurons),from: -1, to: 1
//)
//var w2 = Matrix.random(
//	size: MatrixSize(rows: nNeurons, columns : nNeurons),from: -1, to: 1
//)
//var w3 = Matrix.random(
//	size: MatrixSize(rows: nNeurons, columns : nNeurons),from: -1, to: 1
//)
//var w4 = Matrix.random(
//	size: MatrixSize(rows: nNeurons, columns : y.size().columns),from: -1, to: 1
//)
//
//
//
//
//
//let activation = Swish()
//
//let learningRate = 0.1
//
//
////FOR LOOP!
//for i in 0..<2000 {
//	//FPass
//
//
//	let output0 = activation.run(matrix: (try! Matrix.dot(x, w1)) + 1)
//
//	let output1 = activation.run(matrix: (try! Matrix.dot(output0, w2)) + 1)
//	let output2 = activation.run(matrix: (try! Matrix.dot(output1, w3)) + 1)
//	let output3 =  (try! Matrix.dot(output2, w4) + 1)
//
//
//	let e4 = try! output3 - y
//	let pd4 = try! Matrix.dot(output2.t(), e4)
//	w4 = try! w4 + (pd4 * -learningRate)
//
//	let e3 = try! (activation.der(matrix: output2) * Matrix.dot(e4, w4.t()))
//	let pd3 = try! Matrix.dot(output1.t(), e3)
//	w3 = try! w3 + (pd3 * -learningRate)
//
//	let e2 = try! (activation.der(matrix: output1) * Matrix.dot(e3, w3.t()))
//	let pd2 = try! Matrix.dot(output0.t(), e2)
//	w2 = try! w2 + (pd2 * -learningRate)
//
//	let e1 = try! (activation.der(matrix: output0) * Matrix.dot(e2, w2.t()))
//	let pd1 = try! Matrix.dot(x.t(), e1)
//	w1 = try! w1 + (pd1 * -learningRate)
//
//
//	if (i % 1000 == 0) {
//		print(output3.data())
//
//
//	}
//}
//
//
