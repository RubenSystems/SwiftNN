import Foundation
import UIKit
import CoreGraphics

public struct PixelData {
	var a: UInt8
	var r: UInt8
	var g: UInt8
	var b: UInt8

	public init(value: Double) {
		let newValue = UInt8(value * 255)
		a = 255
		r = newValue
		g = newValue
		b = newValue
	}

}


public func imageFromData(pixels:[PixelData], width: Int, height: Int)-> UIImage? {
	let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
	let bitmapInfo:CGBitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue)
	let bitsPerComponent:Int = 8
	let bitsPerPixel:Int = 32

	assert(pixels.count == Int(width * height))

	var data = pixels // Copy to mutable []
	guard let providerRef = CGDataProvider(data: NSData(bytes: &data, length: data.count * MemoryLayout<PixelData>.size)) else {return nil}

	guard let cgim = CGImage(
			width: width,
			height: height,
			bitsPerComponent: bitsPerComponent,
			bitsPerPixel: bitsPerPixel,
			bytesPerRow: width * MemoryLayout<PixelData>.size,
			space: rgbColorSpace,
			bitmapInfo: bitmapInfo,
			provider: providerRef,
			decode: nil,
			shouldInterpolate: true,
			intent: .defaultIntent
			)
			else { return nil }
	return UIImage(cgImage: cgim)
}

public func resizeImage(image: UIImage, targetSize: CGSize) -> UIImage {
	let size = image.size
	
	let widthRatio  = targetSize.width  / size.width
	let heightRatio = targetSize.height / size.height
	
	// Figure out what our orientation is, and use that to form the rectangle
	var newSize: CGSize
	if(widthRatio > heightRatio) {
		newSize = CGSize(width: size.width * heightRatio, height: size.height * heightRatio)
	} else {
		newSize = CGSize(width: size.width * widthRatio,  height: size.height * widthRatio)
	}
	
	// This is the rect that we've calculated out and this is what is actually used below
	let rect = CGRect(x: 0, y: 0, width: newSize.width, height: newSize.height)
	
	// Actually do the resizing to the rect using the ImageContext stuff
	UIGraphicsBeginImageContextWithOptions(newSize, false, 1.0)
	image.draw(in: rect)
	let newImage = UIGraphicsGetImageFromCurrentImageContext()
	UIGraphicsEndImageContext()
	
	return newImage!
}
