from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from torch.nn.modules import Module
from torch.autograd import Function
from .. import settings

class Matmul(Module):
	def __init__(self, representation_tree):
		super().__init__()
		self.representation_tree = representation_tree

	def forward(self, rhs, *matrix_args):
		return _Matmul.apply(self.representation_tree, self.representation_tree(*matrix_args), rhs, *matrix_args)

class _Matmul(Function):

	@staticmethod
	def forward(ctx, representation_tree, lazy_var, rhs, *matrix_args):
		res = lazy_var._matmul(rhs)
		to_save = [rhs] + list(matrix_args)
		ctx.save_for_backward(*to_save)
		ctx._representation_tree = representation_tree
		if not settings.memory_efficient.on():
			ctx._lazy_var = lazy_var
		return res

	@staticmethod
	def backward(ctx, grad_output):
		rhs = ctx.saved_tensors[0]
		matrix_args = ctx.saved_tensors[1:]
		rhs_shape = rhs.shape

		rhs_grad = None
		arg_grads = [None] * len(matrix_args)

		# input_1 gradient
		if any(ctx.needs_input_grad[1:]):
			rhs = rhs.unsqueeze(-1) if (rhs.ndimension() == 1) else rhs
			grad_output_matrix = grad_output.unsqueeze(-1) if grad_output.ndimension() == 1 else grad_output
			arg_grads = ctx._representation_tree(*matrix_args)._quad_form_derivative(grad_output_matrix, rhs)

		# input_2 gradient
		if ctx.needs_input_grad[0]:
			if hasattr(ctx, "_lazy_var"):
				lazy_var = ctx._lazy_var
			else:
				lazy_var = ctx.representation_tree(*matrix_args)
			rhs_grad = lazy_var._t_matmul(grad_output)
			rhs_grad = rhs_grad.view(rhs_shape)

		return tuple([None, None] + [rhs_grad] + list(arg_grads))