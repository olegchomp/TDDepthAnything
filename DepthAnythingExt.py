import os
import cv2
import numpy as np
from transform import *
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import webbrowser

class DepthAnythingExt:
	def __init__(self, ownerComp):
		# The component to which this extension is attached
		self.ownerComp = ownerComp

		# Create logger and load the TensorRT engine
		engine = parent().par.Enginefile.val
		logger = trt.Logger(trt.Logger.WARNING)
		with open(engine, 'rb') as f, trt.Runtime(logger) as runtime:
			engine = runtime.deserialize_cuda_engine(f.read())
		self.context = engine.create_execution_context()
		self.input_shape = self.context.get_tensor_shape('input')
		self.output_shape = self.context.get_tensor_shape('output')
		self.h_input = cuda.pagelocked_empty(trt.volume(self.input_shape), dtype=np.float32)
		self.h_output = cuda.pagelocked_empty(trt.volume(self.output_shape), dtype=np.float32)
		self.d_input = cuda.mem_alloc(self.h_input.nbytes)
		self.d_output = cuda.mem_alloc(self.h_output.nbytes)
		self.stream = cuda.Stream()

	def load_op_image(self):
		image = op('null1').numpyArray()
		image = (image * 255).astype(np.uint8)
		orig_shape = image.shape[:2]
		image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB) / 255.0
		image = transform({"image": image})["image"]  # C, H, W
		image = image[None]  # B, C, H, W
		return image, orig_shape
	
	def run(self):
		input_image, (orig_h, orig_w) = self.load_op_image()
		
		# Copy the input image to the pagelocked memory
		np.copyto(self.h_input, input_image.ravel())
		
		# Copy the input to the GPU, execute the inference, and copy the output back to the CPU
		cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
		self.context.execute_async_v2(bindings=[int(self.d_input), int(self.d_output)], stream_handle=self.stream.handle)
		cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
		self.stream.synchronize()
		depth = self.h_output
			
		# Process the depth output
		depth = np.reshape(depth, self.output_shape[2:])
		depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0 #  * 65535.0
		depth = depth.astype(np.uint8) # np.uint16
		depth = cv2.resize(depth, (orig_w, orig_h))
		
		if parent().par.Output == 'grayscale':
			image = depth[:, :, np.newaxis].astype(np.uint8) # np.uint16
		else:
			colored_depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
			image = cv2.cvtColor(colored_depth, cv2.COLOR_BGR2RGBA)

		return image
		
	def about(self, endpoint):

		if endpoint == 'Urlg':
			webbrowser.open('https://github.com/olegchomp/TDDepthAnything', new=2)
		if endpoint == 'Urld':
			 webbrowser.open('https://discord.gg/wNW8xkEjrf', new=2)
		if endpoint == 'Urlt':
			webbrowser.open('https://www.youtube.com/vjschool', new=2)
		if endpoint == 'Urla':
			webbrowser.open('https://olegcho.mp/', new=2)
		if endpoint == 'Urldonate':
			webbrowser.open('https://boosty.to/vjschool/donate', new=2)