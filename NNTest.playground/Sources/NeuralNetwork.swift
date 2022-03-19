import Foundation

public protocol Layer {
	
	var output: Matrix? {get set}
	var weights: Matrix? {get set}
	
	
	func size () -> MatrixSize
	
	func activation() -> ActivationFunction
}


public class DenseLayer: Layer {
	
	private let activationFunc : ActivationFunction
	public var weights : Matrix?
	public var output: Matrix?
	
	
	public init(inputSize: Int, outputSize: Int , activation: ActivationFunction) {
		self.activationFunc = activation
		
		self.weights = Matrix.random(
			size: MatrixSize(rows: inputSize, columns : outputSize),from: -1, to: 1
		)
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

	
	public func fit(x: Matrix, y: Matrix, iterations: Int, learningRate: Double) {

		for i in 0..<iterations {
			//FPass
			var prevOutput = x
			for i in 0..<self.layers.count {
				let currentOutput = (try! Matrix.dot(prevOutput, layers[i].weights!)) + 1
				layers[i].output = layers[i].activation().run(matrix: currentOutput)
				
				prevOutput = layers[i].output!
				
			}
			
			var error: Matrix?
			var partialDerivitave: Matrix
			//Backward pass
			for i in (0..<layers.count).reversed() {
				if i == layers.count - 1 {
					error = try! layers[i].output! - y
				} else {

					error = try! (layers[i].activation().der(matrix: layers[i].output!) * Matrix.dot(error!, layers[i + 1].weights!.t()))
				}
				partialDerivitave = try! Matrix.dot((i == 0 ? x : layers[i - 1].output!).t(), error!)
				layers[i].weights = try! layers[i].weights! + (partialDerivitave * -learningRate)
			}

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
