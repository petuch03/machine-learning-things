def backward(t, out_grad):
    dt_dx = 1 - t ** 2
    grad_input = out_grad * dt_dx
    return grad_input


def solution():
    t, out_grad = map(float, input().split())
    print(backward(t, out_grad))


solution()
