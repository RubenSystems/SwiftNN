import Foundation
import Accelerate

public protocol Layer {
	
	var output: Matrix? {get set}
	var weights: Matrix? {get set}
	var bias: [Double]? {get set}
	
	
	func size () -> MatrixSize
	
	func activation() -> ActivationFunction
}


public class DenseLayer: Layer {
	
	private let activationFunc : ActivationFunction
	public var weights : Matrix?
	public var bias : [Double]?
	public var output: Matrix?
	
	
	public init(inputSize: Int, outputSize: Int , activation: ActivationFunction) {
		self.activationFunc = activation
		//Output size is the same as the number of neurons
		self.weights = Matrix.random(size: MatrixSize(rows: inputSize, columns : outputSize),from: -1, to: 1)
		
		self.bias = (0..<outputSize).map { _ in
			Double.random(in: -0.1...0.1)
		}
		
	}

	public func size() -> MatrixSize {
		guard let weights = weights else {
			fatalError("Layer has not yet been compiled")
		}

		return weights.size();
	}
	
	public func activation() -> ActivationFunction {
		return activationFunc
	}
}

public class NeuralNetwork {
	private var layers: [Layer]
	
	public init (layers: [Layer]) {
		self.layers = layers
	}

	
	public func fit(x xFit: Matrix, y yFit: Matrix, epochs: Int, learningRate: Double, batchSize: Int = 64) {
		
		let chunkedX = xFit.formattedData().chunked(into: batchSize).compactMap { Matrix($0) }
		let chunkedY = yFit.formattedData().chunked(into: batchSize).compactMap { Matrix($0) }
		

		var errorAvg = 0.0
		var errorCount = 0.0
		for _ in 0...epochs {
			errorAvg = 0
			errorCount = 0
			for (x, y) in zip(chunkedX, chunkedY){
				//FPass
				var prevOutput = x
				for i in 0..<self.layers.count {
					let currentOutput = ((try! Matrix.dot(prevOutput, layers[i].weights!)) + 1)
					layers[i].output = layers[i].activation().run(matrix: currentOutput)
					prevOutput = layers[i].output!
				}
				
				var error: Matrix?
				var partialDerivitave: Matrix
				//Backward pass
				for i in (0..<layers.count).reversed() {
					if i == layers.count - 1 {
						error = try! layers[i].output! - y
						errorCount += 1
						errorAvg = errorAvg + ((((1 - vDSP.mean(error!.abs().data())) * 100) - errorAvg) / errorCount)
					} else {
						error = try! Matrix.dot(error!, layers[i + 1].weights!.t())
					}
					error = try! layers[i].activation().der(matrix: layers[i].output!) * error!
					partialDerivitave = try! Matrix.dot((i == 0 ? x : layers[i - 1].output!).t(), error!)
					layers[i].weights = try! layers[i].weights! + (partialDerivitave * -learningRate)
					for c in partialDerivitave.formattedData() {
						layers[i].bias = vDSP.subtract(layers[i].bias!, vDSP.multiply(learningRate, c))
					}
				}
			}
			print(errorAvg)
		}
	}

	
	public func predict(x: Matrix) -> Matrix {
		var prevOutput = x
		for i in 0..<self.layers.count {
			let currentOutput = (try! Matrix.dot(prevOutput, layers[i].weights!)) + 1
			prevOutput = layers[i].activation().run(matrix: currentOutput)
		}
		return prevOutput
	}
}


extension Array where Element == Double {
	func average() -> Double {
		return reduce(0.0) {
			return $0 + $1/Double(abs( count))
		}
	}
	
}
