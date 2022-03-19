import Accelerate

public protocol ActivationFunction {
	
	func run(matrix: Matrix) -> Matrix
	
	func der(matrix: Matrix) -> Matrix
}


public class Linear : ActivationFunction {
	
	
	public init (){}
	
	public func run(matrix: Matrix) -> Matrix {
		return matrix
	}
	
	public func der(matrix: Matrix) -> Matrix {
		let result: [Double] = (0..<matrix.size().columns * matrix.size().rows).map { _ in
			1
		}
		
		return Matrix(data: result, size: matrix.size())
	}
}

public class Sigmoid : ActivationFunction {
	public init(){}
	
	public func run(matrix: Matrix) -> Matrix {
		let expVal: [Double] = vForce.exp(vDSP.multiply(-1, matrix.data()))
		let bottomVal : [Double] = vDSP.add(1, expVal)
		return Matrix(data: vDSP.divide(1, bottomVal), size: matrix.size())
	}
	
	public func der(matrix: Matrix) -> Matrix {
		let ranValue = run(matrix: matrix)
		let secondTerm = vDSP.add(1, vDSP.multiply(-1, ranValue.data()))
		let result = vDSP.multiply(ranValue.data(), secondTerm)
		return Matrix(data: result, size: matrix.size())
	}
}

public class Swish : ActivationFunction {
	
	static private var sigmoid : Sigmoid = Sigmoid()
	
	public init () {}
	
	public func run(matrix: Matrix) -> Matrix {
		
		return try! matrix * Swish.sigmoid.run(matrix: matrix)
	}
	
	public func der(matrix: Matrix) -> Matrix {
		
		let secondPart = vDSP.multiply(Swish.sigmoid.run(matrix: matrix).data(), vDSP.add(1, vDSP.multiply(-1, self.run(matrix: matrix).data())))
		return Matrix(data: vDSP.add(self.run(matrix: matrix).data(), secondPart), size: matrix.size())
	}
	

}
