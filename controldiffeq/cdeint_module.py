import torch
import torchdiffeq
import numpy as np 
import time
global h_prime_list
global t_idx
t_idx = 0 
h_prime_list = torch.Tensor()
class VectorField(torch.nn.Module):
    def __init__(self, dX_dt, func):
        """Defines a controlled vector field.

        Arguments:
            dX_dt: As cdeint.
            func: As cdeint.
        """
        super(VectorField, self).__init__()
        if not isinstance(func, torch.nn.Module):
            raise ValueError("func must be a torch.nn.Module.")

        self.dX_dt = dX_dt
        self.func = func
        self.sigmoid = torch.nn.Sigmoid()
    def __call__(self, t, z):
        # import pdb ; pdb.set_trace()
        control_gradient = self.dX_dt(t).float() #1024,69
        # vector_field is of shape (..., hidden_channels, input_channels)
        
        vector_field = self.func(z) # 1024,49,69

        # out is of shape (..., hidden_channels)
        # (The squeezing is necessary to make the matrix-multiply properly batch in all cases)
        out = (vector_field @ control_gradient.unsqueeze(-1)).squeeze(-1) 
        # 1024,23,12 :: 1024,12,20
        # out = self.sigmoid(out)
        # 
        return out

class VectorField_stack(torch.nn.Module):
    def __init__(self, dX_dt, func,final_time,file):
        """Defines a controlled vector field.

        Arguments:
            dX_dt: As cdeint.
            func: As cdeint.
        """
        super(VectorField_stack, self).__init__()
        if not isinstance(func, torch.nn.Module):
            raise ValueError("func must be a torch.nn.Module.")

        self.dX_dt = dX_dt
        self.func = func
        self.final_time = final_time
        self.h_prime_list = h_prime_list
        self.file = file
    def __call__(self, t, z):
        # control_gradient is of shape (..., input_channels)
        # start = time.time()
        # import pdb ; pdb.set_trace()
        
        control_gradient = self.dX_dt(t)
        # vector_field is of shape (..., hidden_channels, input_channels)
        
        vector_field = self.func(z)
        control_gradient = control_gradient.type(vector_field.type())
        # out is of shape (..., hidden_channels)
        # (The squeezing is necessary to make the matrix-multiply properly batch in all cases)
        # import pdb ; pdb.set_trace()
        # vector_field = torch.tensor(vector_field,dtype=torch.float64).to(control_gradient)
        out = (vector_field @ control_gradient.unsqueeze(-1)).squeeze(-1)
        
        
        if self.h_prime_list.shape[0]>0:
            self.h_prime_list = torch.cat([self.h_prime_list,out.unsqueeze(0)],dim=0)
            
        else:
            
            self.h_prime_list = out.unsqueeze(0)
        
        if self.final_time - t <= 0.1: # mali는 181까지 안감
            
            np.save(self.file, self.h_prime_list.cpu().detach().numpy())
        
        return out
class Hardsigmoid(torch.nn.Module):

    def __init__(self):
        super(Hardsigmoid, self).__init__()
        self.act = torch.nn.Hardtanh()

    def forward(self, x):
        return (self.act(x) + 1.0) / 2.0
class RoundFunctionST(torch.autograd.Function):
    """Rounds a tensor whose values are in [0, 1] to a tensor with values in {0, 1}"""

    @staticmethod
    def forward(ctx, input):

        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):

        return grad_output

RoundST = RoundFunctionST.apply

class AttentiveVectorField(torch.nn.Module):
    
    def __init__(self,dX_dt,func_g,X_s,h_prime,time,attention):
        super(AttentiveVectorField, self).__init__()
        if not isinstance(func_g, torch.nn.Module):
            raise ValueError("func must be a torch.nn.Module.")
        self.sigmoid = torch.nn.Sigmoid()
        self.dX_dt = dX_dt
        self.func_g =func_g
        
        self.X_s = X_s  # interpolated_value
        
        self.h_prime=h_prime
        self.timewise = time
        self.attention=attention
        self.t_idx = t_idx

    def __call__(self, t, z):
        Y = self.func_g(z) # attention * x 
        
        control_gradient = self.dX_dt(t).float() # 32,4 # 32,12
        if self.timewise:
            a_t = self.attention[int(np.floor(t.item()))-1,:,:]
        else:
            a_t = self.attention[int(np.floor(t.item()))-1,:,:].squeeze()
        
        Xt = self.dX_dt(t)
        dY_dt_1 = torch.mul(control_gradient,a_t)
        dY_dt_2_0 = torch.mul(torch.mul(a_t,(1-a_t)),Xt)
        # print(self.t_idx)
        dY_dt_2 = torch.mul(dY_dt_2_0,torch.Tensor(self.h_prime[self.t_idx,:,:]).to(dY_dt_2_0.device))
        dY_dt = (dY_dt_1+dY_dt_2).float()
         
        
        out = (Y@dY_dt.unsqueeze(-1)).squeeze(-1)
        self.t_idx +=1
        # import pdb ; pdb.set_trace()
        if self.t_idx ==self.h_prime.shape[0]-1:
            self.t_idx = 0 
        return out

  
def cdeint(dX_dt, z0, func, t, adjoint=True, **kwargs):
    r"""Solves a system of controlled differential equations.

    Solves the controlled problem:
    ```
    z_t = z_{t_0} + \int_{t_0}^t f(z_s)dX_s
    ```
    where z is a tensor of any shape, and X is some controlling signal.

    Arguments:
        dX_dt: The control. This should be a callable. It will be evaluated with a scalar tensor with values
            approximately in [t[0], t[-1]]. (In practice variable step size solvers will often go a little bit outside
            this range as well.) Then dX_dt should return a tensor of shape (..., input_channels), where input_channels
            is some number of channels and the '...' is some number of batch dimensions.
        z0: The initial state of the solution. It should have shape (..., hidden_channels), where '...' is some number
            of batch dimensions.
        func: Should be an instance of `torch.nn.Module`. Describes the vector field f(z). Will be called with a tensor
            z of shape (..., hidden_channels), and should return a tensor of shape
            (..., hidden_channels, input_channels), where hidden_channels and input_channels are integers defined by the
            `hidden_shape` and `dX_dt` arguments as above. The '...' corresponds to some number of batch dimensions.
        t: a one dimensional tensor describing the times to range of times to integrate over and output the results at.
            The initial time will be t[0] and the final time will be t[-1].
        adjoint: A boolean; whether to use the adjoint method to backpropagate.
        **kwargs: Any additional kwargs to pass to the odeint solver of torchdiffeq. Note that empirically, the solvers
            that seem to work best are dopri5, euler, midpoint, rk4. Avoid all three Adams methods.

    Returns:
        The value of each z_{t_i} of the solution to the CDE z_t = z_{t_0} + \int_0^t f(z_s)dX_s, where t_i = t[i]. This
        will be a tensor of shape (len(t), ..., hidden_channels).
    """
    # import pdb ; pdb.set_trace()
    control_gradient = dX_dt(torch.zeros(1, dtype=z0.dtype, device=z0.device))
    # if control_gradient.shape[:-1] != z0.shape[:-1]:
    #     raise ValueError("dX_dt did not return a tensor with the same number of batch dimensions as z0. dX_dt returned "
    #                      "shape {} (meaning {} batch dimensions)), whilst z0 has shape {} (meaning {} batch "
    #                      "dimensions)."
    #                      "".format(tuple(control_gradient.shape), tuple(control_gradient.shape[:-1]), tuple(z0.shape),
    #                                tuple(z0.shape[:-1])))

    # vector_field = func(z0)
    # if vector_field.shape[:-2] != z0.shape[:-1]:
    #     raise ValueError("func did not return a tensor with the same number of batch dimensions as z0. func returned "
    #                      "shape {} (meaning {} batch dimensions)), whilst z0 has shape {} (meaning {} batch"
    #                      " dimensions)."
    #                      "".format(tuple(vector_field.shape), tuple(vector_field.shape[:-2]), tuple(z0.shape),
    #                                tuple(z0.shape[:-1])))
    # if vector_field.size(-2) != z0.shape[-1]:
    #     raise ValueError("func did not return a tensor with the same number of hidden channels as z0. func returned "
    #                      "shape {} (meaning {} channels), whilst z0 has shape {} (meaning {} channels)."
    #                      "".format(tuple(vector_field.shape), vector_field.size(-2), tuple(z0.shape),
    #                                z0.shape.size(-1)))
    # if vector_field.size(-1) != control_gradient.size(-1):
    #     raise ValueError("func did not return a tensor with the same number of input channels as dX_dt returned. "
    #                      "func returned shape {} (meaning {} channels), whilst dX_dt returned shape {} (meaning {}"
    #                      " channels)."
    #                      "".format(tuple(vector_field.shape), vector_field.size(-1), tuple(control_gradient.shape),
    #                                control_gradient.size(-1)))
    # if control_gradient.requires_grad and adjoint:
    #     raise ValueError("Gradients do not backpropagate through the control with adjoint=True. (This is a limitation "
    #                      "of the underlying torchdiffeq library.)")

    odeint = torchdiffeq.odeint_adjoint if adjoint else torchdiffeq.odeint
    vector_field = VectorField(dX_dt=dX_dt, func=func)

    out = odeint(func=vector_field, y0=z0, t=t, **kwargs)

    return out

def cdeint_final(dX_dt,attention,zz0,X_s, func_f,func_g, t,timewise,adjoint=True , **kwargs):

    z0,y0 = zz0
    control_gradient = dX_dt(torch.zeros(1, dtype=y0.dtype, device=y0.device))
    # if control_gradient.shape[:-1] != z0.shape[1:2]:
    #     raise ValueError("dX_dt did not return a tensor with the same number of batch dimensions as z0. dX_dt returned "
    #                      "shape {} (meaning {} batch dimensions)), whilst z0 has shape {} (meaning {} batch "
    #                      "dimensions)."
    #                      "".format(tuple(control_gradient.shape), tuple(control_gradient.shape[:-1]), tuple(z0.shape),
    #                                tuple(z0.shape[:-1])))
    
    vector_field = func_g(y0) 
    #  z0 : shape 57, 1024, 49 
    # vector_field shape 57, 1024, 49,69 
    # if vector_field.shape[1:-1] != z0.shape[1:]:
    #     raise ValueError("func did not return a tensor with the same number of batch dimensions as z0. func returned "
    #                      "shape {} (meaning {} batch dimensions)), whilst z0 has shape {} (meaning {} batch"
    #                      " dimensions)."
    #                      "".format(tuple(vector_field.shape), tuple(vector_field.shape[:-2]), tuple(z0.shape),
    #                                tuple(z0.shape[:-1])))
    # # if vector_field.size(-2) != z0.shape[-1]:
    #     raise ValueError("func did not return a tensor with the same number of hidden channels as z0. func returned "
    #                      "shape {} (meaning {} channels), whilst z0 has shape {} (meaning {} channels)."
    #                      "".format(tuple(vector_field.shape), vector_field.size(-2), tuple(z0.shape),
    #                                z0.shape.size(-1)))
    # if vector_field.size(-1) != control_gradient.size(-1):
    #     raise ValueError("func did not return a tensor with the same number of input channels as dX_dt returned. "
    #                      "func returned shape {} (meaning {} channels), whilst dX_dt returned shape {} (meaning {}"
    #                      " channels)."
    #                      "".format(tuple(vector_field.shape), vector_field.size(-1), tuple(control_gradient.shape),
    #                                control_gradient.size(-1)))
    # if control_gradient.requires_grad and adjoint:
    #     raise ValueError("Gradients do not backpropagate through the control with adjoint=True. (This is a limitation "
    #                      "of the underlying torchdiffeq library.)")
    # import pdb ; pdb.set_trace()
    odeint = torchdiffeq.odeint_adjoint if adjoint else torchdiffeq.odeint
    # vector_field = AttentiveTimeVectorField(dX_dt=dX_dt, func_f=func_f,func_g =func_g,X_s=X_s,linear_f=linear_f,atten_in=atten_in)
    vector_field = AttentiveVectorField(dX_dt=dX_dt,func_f=func_f, func_g =func_g,X_s=X_s,time=timewise,attention = attention)

    # import pdb ; pdb.set_trace()
    out = odeint(func=vector_field.cuda(), y0=zz0, t=t, **kwargs)
    return out






def ancde_bottom(dX_dt, z0, func, t,file, adjoint=True, **kwargs):
    r"""Solves a system of controlled differential equations.

    Solves the controlled problem:
    ```
    z_t = z_{t_0} + \int_{t_0}^t f(z_s)dX_s
    ```
    where z is a tensor of any shape, and X is some controlling signal.

    Arguments:
        dX_dt: The control. This should be a callable. It will be evaluated with a scalar tensor with values
            approximately in [t[0], t[-1]]. (In practice variable step size solvers will often go a little bit outside
            this range as well.) Then dX_dt should return a tensor of shape (..., input_channels), where input_channels
            is some number of channels and the '...' is some number of batch dimensions.
        z0: The initial state of the solution. It should have shape (..., hidden_channels), where '...' is some number
            of batch dimensions.
        func: Should be an instance of `torch.nn.Module`. Describes the vector field f(z). Will be called with a tensor
            z of shape (..., hidden_channels), and should return a tensor of shape
            (..., hidden_channels, input_channels), where hidden_channels and input_channels are integers defined by the
            `hidden_shape` and `dX_dt` arguments as above. The '...' corresponds to some number of batch dimensions.
        t: a one dimensional tensor describing the times to range of times to integrate over and output the results at.
            The initial time will be t[0] and the final time will be t[-1].
        adjoint: A boolean; whether to use the adjoint method to backpropagate.
        **kwargs: Any additional kwargs to pass to the odeint solver of torchdiffeq. Note that empirically, the solvers
            that seem to work best are dopri5, euler, midpoint, rk4. Avoid all three Adams methods.

    Returns:
        The value of each z_{t_i} of the solution to the CDE z_t = z_{t_0} + \int_0^t f(z_s)dX_s, where t_i = t[i]. This
        will be a tensor of shape (len(t), ..., hidden_channels).
    """
    if 'method' not in kwargs:
        kwargs['method'] = 'rk4'
    # import pdb; pdb.set_trace()        
    if kwargs['method'] == 'sym12async':
        if 'atol' not in kwargs:
            kwargs['atol'] = 1e-2
        if 'rtol' not in kwargs:
            kwargs['rtol'] = 1e-3

    elif kwargs['method'] == 'dopri5':
        if 'atol' not in kwargs:
            kwargs['atol'] = 1e-12
        if 'rtol' not in kwargs:
            kwargs['rtol'] = 1e-6

    elif kwargs['method'] == 'rk4':
        if 'options' not in kwargs:
            kwargs['options'] = {}
        options = kwargs['options']
        if 'step_size' not in options and 'grid_constructor' not in options:
            time_diffs = 1.0
            options['step_size'] = time_diffs
    
    control_gradient = dX_dt(torch.zeros(1, dtype=z0.dtype, device=z0.device))
    if control_gradient.shape[:-1] != z0.shape[:-1]:
        raise ValueError("dX_dt did not return a tensor with the same number of batch dimensions as z0. dX_dt returned "
                         "shape {} (meaning {} batch dimensions)), whilst z0 has shape {} (meaning {} batch "
                         "dimensions)."
                         "".format(tuple(control_gradient.shape), tuple(control_gradient.shape[:-1]), tuple(z0.shape),
                                   tuple(z0.shape[:-1])))
    vector_field = func(z0)
    if vector_field.shape[:-2] != z0.shape[:-1]:
        raise ValueError("func did not return a tensor with the same number of batch dimensions as z0. func returned "
                         "shape {} (meaning {} batch dimensions)), whilst z0 has shape {} (meaning {} batch"
                         " dimensions)."
                         "".format(tuple(vector_field.shape), tuple(vector_field.shape[:-2]), tuple(z0.shape),
                                   tuple(z0.shape[:-1])))
    if vector_field.size(-2) != z0.shape[-1]:
        raise ValueError("func did not return a tensor with the same number of hidden channels as z0. func returned "
                         "shape {} (meaning {} channels), whilst z0 has shape {} (meaning {} channels)."
                         "".format(tuple(vector_field.shape), vector_field.size(-2), tuple(z0.shape),
                                   z0.shape.size(-1)))
    if vector_field.size(-1) != control_gradient.size(-1):
        raise ValueError("func did not return a tensor with the same number of input channels as dX_dt returned. "
                         "func returned shape {} (meaning {} channels), whilst dX_dt returned shape {} (meaning {}"
                         " channels)."
                         "".format(tuple(vector_field.shape), vector_field.size(-1), tuple(control_gradient.shape),
                                   control_gradient.size(-1)))
    if control_gradient.requires_grad and adjoint:
        raise ValueError("Gradients do not backpropagate through the control with adjoint=True. (This is a limitation "
                         "of the underlying torchdiffeq library.)")

    odeint = torchdiffeq.odeint_adjoint if adjoint else torchdiffeq.odeint

    vector_field = VectorField_stack(dX_dt=dX_dt, func=func,final_time=t[-1],file=file)
    if kwargs['method'] == 'sym12async':
        #import pdb ; pdb.set_trace()
        for i in range(len(t)-1):
            kwargs["t0"]  = t[i]
            kwargs["t1"]  = t[i+1]
            output = odesolve(vector_field, z0, kwargs)
            
            if i ==0 :
                real_output = torch.cat((z0.unsqueeze(0),output.unsqueeze(0)), dim=0)
                z0 = output
            else:
                real_output = torch.cat((real_output,output.unsqueeze(0)), dim=0)
                z0 = output
    else:
        out = odeint(func=vector_field, y0=z0, t=t, **kwargs)
        real_output = out
    

    return real_output



def ancde(dX_dt,attention,z0,X_s, func_g,h_prime, t,timewise,adjoint=True , **kwargs):

    
    control_gradient = dX_dt(torch.zeros(1, dtype=z0.dtype, device=z0.device))
    # if control_gradient.shape[:-1] != z0.shape[1:2]:
    #     raise ValueError("dX_dt did not return a tensor with the same number of batch dimensions as z0. dX_dt returned "
    #                      "shape {} (meaning {} batch dimensions)), whilst z0 has shape {} (meaning {} batch "
    #                      "dimensions)."
    #                      "".format(tuple(control_gradient.shape), tuple(control_gradient.shape[:-1]), tuple(z0.shape),
    #                                tuple(z0.shape[:-1])))
    
    vector_field = func_g(z0) 
    
    odeint = torchdiffeq.odeint_adjoint if adjoint else torchdiffeq.odeint
    # vector_field = AttentiveTimeVectorField(dX_dt=dX_dt, func_f=func_f,func_g =func_g,X_s=X_s,linear_f=linear_f,atten_in=atten_in)
    vector_field = AttentiveVectorField(dX_dt=dX_dt, func_g =func_g,X_s=X_s,h_prime=h_prime,time=timewise,attention = attention)

    # import pdb ; pdb.set_trace()
    out = odeint(func=vector_field.cuda(), y0=z0, t=t, **kwargs)
    return out

