import Foundation
import Accelerate




let mnist = Bundle.main.decode(Dataset.self, from: "mnist.json")



let layer1 = DenseLayer(inputSize: 784, outputSize: 128, activation: ReLU())
let layer2 =  DenseLayer(inputSize: 128, outputSize: 10, activation: Linear())

let network = NeuralNetwork(layers: [
	layer1, layer2
])

let generator = NeuralNetwork(layers: [
	layer2
])

network.fit(
	x: mnist.xtrain,
	y: mnist.ytrain,
	epochs: 10,
	learningRate: 0.1,
	batchSize: 64
)

let index = 0
let value = mnist.xtrain.formattedData()[index]
let answer = mnist.ytrain.formattedData()[index]


let imageData = network.predict(x: Matrix([value])).formattedData()


//
//let generatedImageData = imageData.data().map {
//	PixelData(value: $0)
//}
//
//
//resizeImage(image: imageFromData(pixels: generatedImageData, width: 28, height: 28)!, targetSize: CGSize(width: 128, height: 128))

//let colorSpace = CGColorSpaceCreateDeviceGray()
//let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue);
//guard let bitmap = CGContext(data: nil, width: imageData.size().columns, height: imageData.size().rows, bitsPerComponent: 8, bytesPerRow: 0, space: colorSpace, bitmapInfo: bitmapInfo.rawValue) else {fatalError()}
//
//var bitmapData = UnsafeMutablePointer<UInt32>( OpaquePointer(bitmap.data) )
//
//for i in 0...imageData.data().count {
//	bitmapData![i] = UInt32(imageData.data()[i])
//}
//
//let imageRef = bitmap.makeImage()!
//let image = UIImage(cgImage: imageRef)
























//[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
//print(network.predict(x: x).formattedData())



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
