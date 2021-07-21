import torch
import torch.nn as nn
from torchdiffeq._impl.odeint import SOLVERS, odeint
from torchdiffeq._impl.misc import _flat_to_shape, _check_inputs

"""
class OdeintDFAMethod(torch.autograd.Function):
    @staticmethod
    def forward(ctx, device, random_matrix, shapes, func, y0, t, rtol=1e-7, atol=1e-9, method=None, options=None, event_fn=None, *dfa_params):
        ctx.shapes = shapes
        ctx.func = func
        ctx.event_mode = event_fn is not None
        ctx.random_matrix = random_matrix
        ctx.y0 = y0
        ctx.device = device

        with torch.no_grad():
        #with torch.enable_grad():
            func.base_func.y_t = []
            ans = odeint(func, y0, t, rtol=rtol, atol=atol, method=method, options=options, event_fn=event_fn)

            if event_fn is None:
                y = ans
            else:
                event_t, y = ans
                ctx.event_t = event_t

        ctx.save_for_backward(t, y, *dfa_params)
        return ans

    @staticmethod
    def backward(ctx, *grad_y):
        with torch.no_grad():
            func = ctx.func.base_func
            y_t = ctx.func.y_t

            random_matrix = ctx.random_matrix
            t, y, *dfa_params = ctx.saved_tensors
            device = ctx.device
            dfa_params = tuple(dfa_params)


            # Backprop as if integrating up to event time.
            # Does NOT backpropagate through the event time.
            event_mode = ctx.event_mode
            if event_mode:
                event_t = ctx.event_t
                _t = t
                t = torch.cat([t[0].reshape(-1), event_t.reshape(-1)])
                grad_y = grad_y[1]
            else:
                grad_y = grad_y[0]
            y0 = grad_y[0]
            grad_y = grad_y[-1]

            nabla = torch.unsqueeze(torch.mm(grad_y.squeeze(), random_matrix), dim=1)
            nabla.to(device)
            with torch.autograd.set_detect_anomaly(True):
                for y_i in y_t:
                    y_i.backward(nabla)
        return None, None, None, None, None, None, None, None, None, None, None, *dfa_params"""


def odeint_dfa2(device, func, y0, t, *, rtol=1e-7, atol=1e-9, method=None, options=None, event_fn=None):
    if not isinstance(func, nn.Module):
        raise ValueError('func must be an instance of nn.Module to specify the dfa parameters.')

    dfa_params = tuple(find_parameters(func))

    # Filter params that don't require gradients.
    dfa_params = tuple(p for p in dfa_params if p.requires_grad)

    # Convert to flattened state.
    shapes, func, y0, t, rtol, atol, method, options, event_fn, decreasing_time = _check_inputs(func, y0, t, rtol, atol, method, options, event_fn, SOLVERS)

    func.base_func.tab_t = []
    func.base_func.y_t = []
    ans = odeint(func, y0, t, rtol=rtol, atol=atol, method=method, options=options)

    if event_fn is None:
        solution = ans
    else:
        event_t, solution = ans
        event_t = event_t.to(t)
        if decreasing_time:
            event_t = -event_t

    if shapes is not None:
        solution = _flat_to_shape(solution, (len(t),), shapes)

    if event_fn is None:
        return solution
    else:
        return event_t, solution

def odeint_dfa(random_mat, device, func, y0, t, *, rtol=1e-7, atol=1e-9, method=None, options=None, event_fn=None):
    if not isinstance(func, nn.Module):
        raise ValueError('func must be an instance of nn.Module to specify the dfa parameters.')

    dfa_params = tuple(find_parameters(func))

    # Filter params that don't require gradients.
    dfa_params = tuple(p for p in dfa_params if p.requires_grad)

    # Convert to flattened state.
    shapes, func, y0, t, rtol, atol, method, options, event_fn, decreasing_time = _check_inputs(func, y0, t, rtol, atol, method, options, event_fn, SOLVERS)

    ans = OdeintDFAMethod.apply(device, random_mat, shapes, func, y0, t, rtol, atol, method, options, event_fn, *dfa_params)

    if event_fn is None:
        solution = ans
    else:
        event_t, solution = ans
        event_t = event_t.to(t)
        if decreasing_time:
            event_t = -event_t

    if shapes is not None:
        solution = _flat_to_shape(solution, (len(t),), shapes)

    if event_fn is None:
        return solution
    else:
        return event_t, solution


def find_parameters(module):
    assert isinstance(module, nn.Module)

    # If called within DataParallel, parameters won't appear in module.parameters().
    if getattr(module, '_is_replica', False):

        def find_tensor_attributes(module):
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v) and v.requires_grad]
            return tuples

        gen = module._named_members(get_members_fn=find_tensor_attributes)
        return [param for _, param in gen]
    else:
        return list(module.parameters())


