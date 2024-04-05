import tensorrt as trt
import torch
import numpy as np
import torchvision.transforms as transforms
import webbrowser

class DepthAnythingExt:
	def __init__(self, ownerComp):
		self.ownerComp = ownerComp
		self.ownerComp.par.Dimensions = ''
		"""Initialize TensorRT plugins, engine and conetxt."""
		self.trt_path = self.ownerComp.par.Enginefile.val
		self.device = "cuda"
		self.trt_logger = trt.Logger(trt.Logger.INFO)
		try:
			self.engine = self._load_engine()
			self.get_dimensions(self.engine)
			self.context = self.engine.create_execution_context()
			self.stream = torch.cuda.current_stream(device=self.device)
		except Exception as e:
			debug(e)

		self.source = op('null1')
		self.trt_input = torch.zeros((self.source.height, self.source.width), device=self.device)
		self.trt_output = torch.zeros((self.source.height, self.source.width), device=self.device)
		self.to_tensor = TopArrayInterface(self.source)
		self.normalize = transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))

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
		self.stream.synchronize()
			
	def run(self, scriptOp):

		if self.ownerComp.par.Enginefile.val != '' and self.ownerComp.par.Venvpath.val != '':
			self.to_tensor.update(self.stream.cuda_stream)
			self.trt_input = torch.as_tensor(self.to_tensor, device=self.device)
			self.trt_input = self.normalize(self.trt_input[1:, :, :]).ravel()

			self.infer(self.trt_input, self.trt_output)
			
			if self.ownerComp.par.Normalize == 'normal':
				tensor_min = self.trt_output.min()
				tensor_max = self.trt_output.max()
				self.trt_output = (self.trt_output - tensor_min) / (tensor_max - tensor_min)


			output = TopCUDAInterface(
				self.source.width,
				self.source.height,
				1,
				np.float32
			)
			
			scriptOp.copyCUDAMemory(self.trt_output.data_ptr(), output.size,  output.mem_shape)
	
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

class TopArrayInterface:
	def __init__(self, top, stream=0):
		self.top = top
		mem = top.cudaMemory(stream=stream)
		self.w, self.h = mem.shape.width, mem.shape.height
		self.num_comps = mem.shape.numComps
		self.dtype = mem.shape.dataType
		shape = (mem.shape.numComps, self.h, self.w)
		dtype_info = {'descr': [('', '<f4')], 'num_bytes': 4}
		dtype_descr = dtype_info['descr']
		num_bytes = dtype_info['num_bytes']
		num_bytes_px = num_bytes * mem.shape.numComps
		
		self.__cuda_array_interface__ = {
			"version": 3,
			"shape": shape,
			"typestr": dtype_descr[0][1],
			"descr": dtype_descr,
			"stream": stream,
			"strides": (num_bytes, num_bytes_px * self.w, num_bytes_px),
			"data": (mem.ptr, False),
		}

	def update(self, stream=0):

		mem = self.top.cudaMemory(stream=stream)
		self.__cuda_array_interface__['stream'] = stream
		self.__cuda_array_interface__['data'] = (mem.ptr, False)
		return