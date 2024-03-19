import os
import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import webbrowser
import transform

class DepthAnythingExt:
	def __init__(self, ownerComp):
		# The component to which this extension is attached
		self.ownerComp = ownerComp

		# Create logger and load the TensorRT engine
		engine = parent().par.Enginefile.val
		logger = trt.Logger(trt.Logger.WARNING)
		with open(engine, 'rb') as f, trt.Runtime(logger) as runtime:
			engine = runtime.deserialize_cuda_engine(f.read())
			
		context = engine.create_execution_context()
		input_shape = context.get_tensor_shape('input')
		output_shape = context.get_tensor_shape('output')
		h_input = cuda.pagelocked_empty(trt.volume(input_shape), dtype=np.float32)
		h_output = cuda.pagelocked_empty(trt.volume(output_shape), dtype=np.float32)
		d_input = cuda.mem_alloc(h_input.nbytes)
		d_output = cuda.mem_alloc(h_output.nbytes)
		stream = cuda.Stream()
		self.init_scripts = context, input_shape, output_shape, h_input, h_output, d_input, d_output, stream

	def load_op_image(self):
		image = op('null1').numpyArray()
		image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
		orig_shape = image.shape[:2]
		image = transform.transform({"image": image})["image"]  # C, H, W
		image = image[None]  # B, C, H, W
		return image, orig_shape
	
	def run(self):
		context, input_shape, output_shape, h_input, h_output, d_input, d_output, stream = self.init_scripts
		input_image, (orig_h, orig_w) = self.load_op_image()
		
		# Copy the input image to the pagelocked memory
		np.copyto(h_input, input_image.ravel())
		
		# Copy the input to the GPU, execute the inference, and copy the output back to the CPU
		cuda.memcpy_htod_async(d_input, h_input, stream)
		context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
		cuda.memcpy_dtoh_async(h_output, d_output, stream)
		depth = h_output

		# Process the depth output
		depth = np.reshape(depth, output_shape[2:])
		depth = depth.astype(np.float32) # np.uint16
		depth = (depth - depth.min()) / (depth.max() - depth.min())

		image = depth[:, :, np.newaxis].astype(np.float32) # np.uint16

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