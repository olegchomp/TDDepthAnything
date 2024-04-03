import tensorrt as trt
import torch
import cupy as cp
import numpy as np
import webbrowser

class DepthAnythingExt:
	def __init__(self, ownerComp):
		"""Initialize TensorRT plugins, engine and conetxt."""
		self.ownerComp = ownerComp
		self.trt_path = parent().par.Enginefile.val
		self.device = "cuda"
		self.trt_logger = trt.Logger(trt.Logger.INFO)
		self.ownerComp.par.Dimensions = ''
		try:
			self.engine = self._load_engine()
			self.get_dimensions(self.engine)
			self.context = self.engine.create_execution_context()
			self.stream = torch.cuda.current_stream(device=self.device)
		except Exception as e:
			print(e)

		self.cpCudaMem = cp.cuda.memory

	def get_dimensions(self, engine):
		shapes = []
		for binding in range(engine.num_bindings):
			shape = engine.get_binding_shape(binding)
			shapes.append(shape)
		dimensions = shapes[0][2:]
		self.ownerComp.par.Dimensions = f"{dimensions[1]}x{dimensions[0]}"
		op('fit2').par.resolutionw = dimensions[1]
		op('fit2').par.resolutionh = dimensions[0]

	def _load_engine(self):
		"""Load TensorRT engine."""
		TRTbin = self.trt_path
		with open(TRTbin, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
			return runtime.deserialize_cuda_engine(f.read())
	
	def infer(self, img, output):
		"""Run inference on TensorRT engine."""
		self.bindings = [img.data_ptr()] + [output.data_ptr()]
		self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.cuda_stream)
		#self.stream.synchronize()
			
	def run(self):
		cpCudaMem = self.cpCudaMem

		source = op('null1')
		topCudaMem = source.cudaMemory()

		shape = (source.height, source.width, 4)
		dType = cp.float32
		
		cpMemoryPtr = cpCudaMem.MemoryPointer(cpCudaMem.UnownedMemory(
						topCudaMem.ptr, topCudaMem.size, topCudaMem),
						0)
						
		frameGPU = cp.ndarray(shape, dType, cpMemoryPtr)
		mean = cp.asarray([0.485, 0.456, 0.406], dtype=cp.float32)
		std = cp.asarray([0.229, 0.224, 0.225], dtype=cp.float32)
		normalized_image = (frameGPU[..., :3] - mean) / std

		trt_input = torch.as_tensor(normalized_image, device=self.device)
		trt_input = trt_input.permute(2,0,1) * 2 - 1
		trt_input = trt_input.ravel()
		trt_output = torch.zeros((source.height, source.width), device=self.device)
		self.infer(trt_input, trt_output)

		trt_output = trt_output / 255
		trt_output = trt_output.data_ptr()


		output = TopCUDAInterface(
			source.width,
			source.height,
			1,
			np.float32
		)
		
		return trt_output, output
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
class TopCUDAInterface:
	def __init__(self, width, height, num_comps, dtype):
		self.mem_shape = CUDAMemoryShape()
		self.mem_shape.width = width
		self.mem_shape.height = height
		self.mem_shape.numComps = num_comps
		self.mem_shape.dataType = dtype
		self.bytes_per_comp = np.dtype(dtype).itemsize
		self.size = width * height * num_comps * self.bytes_per_comp