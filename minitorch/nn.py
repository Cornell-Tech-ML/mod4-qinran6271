from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    # raise NotImplementedError("Need to implement for Task 4.3")

    # # Calculate new height and width
    new_height = height // kh
    new_width = width // kw

    # Reshape input tensor to split into patches
    reshaped = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)

    # Move kernel dimensions (kh, kw) to the last dimension
    tiled = reshaped.permute(0, 1, 2, 4, 3, 5)

    # Combine kernel dimensions into one
    tiled = tiled.contiguous().view(batch, channel, new_height, new_width, kh * kw)

    return tiled, new_height, new_width


# TODO: Implement for Task 4.3.
def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Average pooling operation"""
    tiled, new_height, new_width = tile(input, kernel)
    return tiled.mean(dim=-1).view(
        input.shape[0], input.shape[1], new_height, new_width
    )


# max, softmax, and log softmax on tensors as well as the dropout and max-pooling operations
max_reduce = FastOps.reduce(operators.max, -float("inf"))


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor"""
    out = max_reduce(input, dim)
    return out == input


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Forward function for max"""
        d = int(dim.item())
        ctx.save_for_backward(input, d)
        return max_reduce(input, d)

    @staticmethod
    def backward(ctx: Context, d_output: Tensor) -> Tuple[Tensor, float]:
        """Backward function for max"""
        input, d = ctx.saved_values
        d_input = argmax(input, d) * d_output
        return d_input, 0.0


def max(input: Tensor, dim: int) -> Tensor:
    """Apply max reduction"""
    return Max.apply(input, input._ensure_tensor(dim))


exp_map = FastOps.map(operators.exp)


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax as a tensor"""
    input_exp = input.exp()
    return input_exp / input_exp.sum(dim)


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax as a tensor"""
    # Find the maximum value along the specified dimension for numerical stability
    max_values = max(input, dim)

    # Subtract the max value for numerical stability
    stabilized_input = input - max_values

    # Compute the log of the sum of exponentials
    log_sum_exp = (stabilized_input.exp()).sum(dim).log()

    # Compute log softmax
    return stabilized_input - log_sum_exp


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled max pooling 2D

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor : pooled tensor

    """
    tiled, new_height, new_width = tile(input, kernel)
    return max(tiled, dim=-1).view(
        input.shape[0], input.shape[1], new_height, new_width
    )


def dropout(input: Tensor, p: float, ignore: bool = False) -> Tensor:
    """Dropout positions based on random noise, include an argument to turn off"""
    if ignore:
        return input
    return input * (rand(input.shape) > p)
