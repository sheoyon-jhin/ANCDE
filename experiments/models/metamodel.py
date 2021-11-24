import pathlib
import sys
import torch
import math
here = pathlib.Path(__file__).resolve().parent
sys.path.append(str(here / '..' / '..'))

import controldiffeq
import numpy as np
class NeuralCDE(torch.nn.Module):
    """A Neural CDE model. Provides a wrapper around the lower-level cdeint function, to get a flexible Neural CDE
    model.

    Specifically, considering the CDE
    ```
    z_t = z_{t_0} + \int_{t_0}^t f(z_s)dX_s
    ```
    where X is determined by the data, and given some terminal time t_N, then this model first computes z_{t_N}, then
    performs a linear function on it, and then outputs the result.

    It's known that linear functions on CDEs are universal approximators, so this is a very general type of model.
    """
    def __init__(self, func, input_channels, hidden_channels, output_channels, initial=True):
        """
        Arguments:
            func: As cdeint.
            input_channels: How many channels there are in the input.
            hidden_channels: The number of hidden channels, i.e. the size of z_t.
            output_channels: How many channels to perform a linear map to at the end.
            initial: Whether to automatically construct the initial value from data (in which case z0 must not be passed
                during forward()), or to use the one supplied during forward (in which case z0 must be passed during
                forward()).
        """
        if isinstance(func, ContinuousRNNConverter):  # ugly hack
            hidden_channels = hidden_channels + input_channels

        super(NeuralCDE, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        
        self.func = func
        self.initial = initial
        if initial and not isinstance(func, ContinuousRNNConverter):  # very ugly hack
            self.initial_network = torch.nn.Linear(input_channels, hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, output_channels)

    def extra_repr(self):
        return "input_channels={}, hidden_channels={}, output_channels={}, initial={}" \
               "".format(self.input_channels, self.hidden_channels, self.output_channels, self.initial)

    def forward(self, times, coeffs, final_index, z0=None, stream=False, **kwargs):
        """
        Arguments:
            times: The times of the observations for the input path X, e.g. as passed as an argument to
                `controldiffeq.natural_cubic_spline_coeffs`.
            coeffs: The coefficients describing the input path X, e.g. as returned by
                `controldiffeq.natural_cubic_spline_coeffs`.
            final_index: Each batch element may have a different final time. This defines the index within the tensor
                `times` of where the final time for each batch element is.
            z0: See the 'initial' argument to __init__.
            stream: Whether to return the result of the Neural CDE model at all times (True), or just the final time
                (False). Defaults to just the final time. The `final_index` argument is ignored if stream is True.
            **kwargs: Will be passed to cdeint.

        Returns:
            If stream is False, then this will return the terminal time z_T. If stream is True, then this will return
            all intermediate times z_t, for those t for which there was data.
        """

        # Extract the sizes of the batch dimensions from the coefficients

        coeff, _, _, _ = coeffs
        # in UEA : coeff : 32,181,4
        # in socar : coeff: 32,1205,11
        # in sepsis : coeffs : 1024,71,69
        batch_dims = coeff.shape[:-2] # 32
        if not stream:
            assert batch_dims == final_index.shape, "coeff.shape[:-2] must be the same as final_index.shape. " \
                                                    "coeff.shape[:-2]={}, final_index.shape={}" \
                                                    "".format(batch_dims, final_index.shape)

        cubic_spline = controldiffeq.NaturalCubicSpline(times, coeffs) # interpolated values
        # TODO z0
        if z0 is None:
            assert self.initial, "Was not expecting to be given no value of z0."
            if isinstance(self.func, ContinuousRNNConverter):  # still an ugly hack
                z0 = torch.zeros(*batch_dims, self.hidden_channels, dtype=coeff.dtype, device=coeff.device) # 32,32
            else:
                z0 = self.initial_network(cubic_spline.evaluate(times[0]))
        else:
            assert not self.initial, "Was expecting to be given a value of z0."
            if isinstance(self.func, ContinuousRNNConverter):  # continuing adventures in ugly hacks
                z0_extra = torch.zeros(*batch_dims, self.input_channels, dtype=z0.dtype, device=z0.device)
                z0 = torch.cat([z0_extra, z0], dim=-1)
        # for sepsis : z0 already exist
        # Figure out what times we need to solve for
        if stream:
            t = times
        else:
            # faff around to make sure that we're outputting at all the times we need for final_index.
            # in UEA : final_index : 32
            # in socar : final_index : 32
            # sorted_final_index : 뒤죽박죽인 final index sort,
            # inverse_final_index : sorted_final_index기준으로 final_index의 index 가지고 있음.
            sorted_final_index, inverse_final_index = final_index.unique(sorted=True, return_inverse=True)
            if 0 in sorted_final_index:
                sorted_final_index = sorted_final_index[1:]
                final_index = inverse_final_index
            else:
                final_index = inverse_final_index + 1
            if len(times) - 1 in sorted_final_index:
                sorted_final_index = sorted_final_index[:-1]

            t = torch.cat([times[0].unsqueeze(0), times[sorted_final_index], times[-1].unsqueeze(0)])

        # Switch default solver
        if 'method' not in kwargs:
            kwargs['method'] = 'rk4'
        if kwargs['method'] == 'rk4':
            if 'options' not in kwargs:
                kwargs['options'] = {}
            options = kwargs['options']
            if 'step_size' not in options and 'grid_constructor' not in options:
                time_diffs = times[1:] - times[:-1]
                options['step_size'] = time_diffs.min().item()

        # Actually solve the CDE

        # TODO
        # UEA DATASET z0 : 32,32
        # SOCAR DATASET z0 : 32,32
        # SEPSIS DATASET z0 :1024,49
        
        z_t = controldiffeq.cdeint(dX_dt=cubic_spline.derivative,
                                   z0=z0,
                                   func=self.func,
                                   t=t,
                                   **kwargs)

        
        if stream:
            # z_t is a tensor of shape (times, ..., channels), so change this to (..., times, channels)
            for i in range(len(z_t.shape) - 2, 0, -1):
                z_t = z_t.transpose(0, i)
        else:
            # final_index is a tensor of shape (...)
            # z_t is a tensor of shape (times, ..., channels)
            final_index_indices = final_index.unsqueeze(-1).expand(z_t.shape[1:]).unsqueeze(0)
            z_t = z_t.gather(dim=0, index=final_index_indices).squeeze(0)

        # Linear map and return
        pred_y = self.linear(z_t)


        return pred_y

class Hardsigmoid(torch.nn.Module):

    def __init__(self):
        super(Hardsigmoid, self).__init__()
        self.act = torch.nn.Hardtanh()

    def forward(self, x):
        return ((self.act(x) + 1.0) / 2.0)
        
class RoundFunctionST(torch.autograd.Function):
    """Rounds a tensor whose values are in [0, 1] to a tensor with values in {0, 1}"""

    @staticmethod
    def forward(ctx, input):

        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):

        return grad_output

RoundST = RoundFunctionST.apply
# Note that this relies on the first channel being time
class ContinuousRNNConverter(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, model):
        super(ContinuousRNNConverter, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.model = model

        out_base = torch.zeros(self.input_channels + self.hidden_channels, self.input_channels)
        for i in range(self.input_channels):
            out_base[i, i] = 1
        self.register_buffer('out_base', out_base)

    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}".format(self.input_channels, self.hidden_channels)

    def forward(self, z):
        # z is a tensor of shape (..., input_channels + hidden_channels)
        x = z[..., :self.input_channels]
        h = z[..., self.input_channels:]
        h = h.clamp(-1, 1)
        # model_out is a tensor of shape (..., hidden_channels)
        model_out = self.model(x, h)
        batch_dims = model_out.shape[:-1]
        out = self.out_base.repeat(*batch_dims, 1, 1).clone()
        out[..., self.input_channels:, 0] = model_out
        return out

class ANCDE(torch.nn.Module):
    
    def __init__(self, func,func_g, input_channels, hidden_channels, output_channels,attention_channel,slope_check,soft, timewise,file,initial=True):
        
        if isinstance(func, ContinuousRNNConverter):  # ugly hack
            hidden_channels = hidden_channels + input_channels

        super(ANCDE, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels

        self.func_f = func 
        self.func_g = func_g 
        self.initial = initial
        self.attention_channel = attention_channel
        self.slope_check = slope_check
        self.soft = soft
        self.file = file 
        self.STE = Hardsigmoid()
        self.binarizer = RoundST 
        if initial and not isinstance(func, ContinuousRNNConverter):  # very ugly hack
            self.initial_network = torch.nn.Linear(input_channels, input_channels)

        self.feature_extractor = torch.nn.Linear(input_channels,hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, output_channels) # hidden state -> prediction
        self.time_attention = torch.nn.Linear(input_channels,1)
        self.timewise = timewise
    def extra_repr(self):
        return "input_channels={}, hidden_channels={}, output_channels={}, initial={}" \
               "".format(self.input_channels, self.hidden_channels, self.output_channels, self.initial)

    def forward(self, times, coeffs, final_index,slope, z0=None, stream=False, **kwargs):
        

        # Extract the sizes of the batch dimensions from the coefficients
        
        coeff, _, _, _ = coeffs
        # SEPSIS : 1024,71,69
        batch_dims = coeff.shape[:-2] # 32
        
        if not stream:
            assert batch_dims == final_index.shape, "coeff.shape[:-2] must be the same as final_index.shape. " \
                                                    "coeff.shape[:-2]={}, final_index.shape={}" \
                                                    "".format(batch_dims, final_index.shape)
        # TODO
        cubic_spline = controldiffeq.NaturalCubicSpline(times, coeffs) # interpolated values
        

        if z0 is None:
            assert self.initial, "Was not expecting to be given no value of z0."
            if isinstance(self.func_f, ContinuousRNNConverter):  # still an ugly hack
                z0 = torch.zeros(*batch_dims, self.hidden_channels, dtype=coeff.dtype, device=coeff.device) # 32,32
            else:
                # z0 = self.initial_network(cubic_spline.evaluate(times[0]))  # 32,32,4
                XX = cubic_spline.evaluate(times[0]) # 32,4
                z0 = self.initial_network(XX) # aa0 32,32

        else:
            assert not self.initial, "Was expecting to be given a value of z0."
            if isinstance(self.func_f, ContinuousRNNConverter):  # continuing adventures in ugly hacks

                z0_extra = torch.zeros(*batch_dims, self.input_channels, dtype=z0.dtype, device=z0.device)
                z0 = torch.cat([z0_extra, z0], dim=-1)
            else:
                self.initial_network = None

        # Figure out what times we need to solve for
        if stream:
            t = times
        else:

            sorted_final_index, inverse_final_index = final_index.unique(sorted=True, return_inverse=True)
            if 0 in sorted_final_index:
                sorted_final_index = sorted_final_index[1:]
                final_index = inverse_final_index
            else:
                final_index = inverse_final_index + 1
            if len(times) - 1 in sorted_final_index:
                sorted_final_index = sorted_final_index[:-1]

            t = torch.cat([times[0].unsqueeze(0), times[sorted_final_index], times[-1].unsqueeze(0)])

        # Switch default solver
        if 'method' not in kwargs:
            kwargs['method'] = 'rk4'
        if kwargs['method'] == 'rk4':
            if 'options' not in kwargs:
                kwargs['options'] = {}
            options = kwargs['options']
            if 'step_size' not in options and 'grid_constructor' not in options:
                time_diffs = times[1:] - times[:-1]
                options['step_size'] = time_diffs.min().item()


        sigmoid = torch.nn.Sigmoid()

        self.atten_in = self.hidden_channels
        
        attention = controldiffeq.ancde_bottom(dX_dt=cubic_spline.derivative,
                                   z0=z0,
                                   func=self.func_f,
                                   t=times,
                                   file=self.file,
                                   **kwargs)
        h_prime = np.load(self.file)
        if self.timewise:
            
            attention = self.time_attention(attention)
            # np.save(self.file_timewise, self.h_prime_list.cpu().detach().numpy()) 
            # h_prime = self.time_attention's weight
            h_prime= self.time_attention.weight
        
        if self.soft :
            
            attention = sigmoid(attention)
        else:
            if self.slope_check :
                
                attention = self.STE(slope * attention)
                attention = self.binarizer(attention)
            else :
                
                attention = sigmoid(attention) # sigmoid -> 다른 것 
                attention = self.binarizer(attention)
        
        x0 = cubic_spline.evaluate(times[0])
        a0 =  attention[0,:,:]
        y0 = torch.mul(x0,a0)
        y0 = self.feature_extractor(y0) 
        
        z_t = controldiffeq.ancde(dX_dt=cubic_spline.derivative,
                                   attention=attention,
                                   z0 =y0,
                                   X_s=cubic_spline,
                                   func_g = self.func_g,
                                   h_prime = h_prime,
                                   t=t,
                                   timewise=self.timewise,
                                   **kwargs)
        
        # length = int(z_t.shape[1]/2)
        # z_t = z_t[:,:length,:]

        # fake_z , z_t = zz_t
        if stream:
            # z_t is a tensor of shape (times, ..., channels), so change this to (..., times, channels)
            for i in range(len(z_t.shape) - 2, 0, -1):
                z_t = z_t.transpose(0, i)
        else:
            # final_index is a tensor of shape (...)
            # z_t is a tensor of shape (times, ..., channels)
            final_index_indices = final_index.unsqueeze(-1).expand(z_t.shape[1:]).unsqueeze(0)
            z_t = z_t.gather(dim=0, index=final_index_indices).squeeze(0)

        # Linear map and return
        pred_y = self.linear(z_t)


        return pred_y

class NeuralCDE_forecasting(torch.nn.Module):
    """A Neural CDE model. Provides a wrapper around the lower-level cdeint function, to get a flexible Neural CDE
    model.

    Specifically, considering the CDE
    ```
    z_t = z_{t_0} + \int_{t_0}^t f(z_s)dX_s
    ```
    where X is determined by the data, and given some terminal time t_N, then this model first computes z_{t_N}, then
    performs a linear function on it, and then outputs the result.

    It's known that linear functions on CDEs are universal approximators, so this is a very general type of model.
    """
    def __init__(self, func, input_channels, hidden_channels, output_channels, initial=True):
        """
        Arguments:
            func: As cdeint.
            input_channels: How many channels there are in the input.
            hidden_channels: The number of hidden channels, i.e. the size of z_t.
            output_channels: How many channels to perform a linear map to at the end.
            initial: Whether to automatically construct the initial value from data (in which case z0 must not be passed
                during forward()), or to use the one supplied during forward (in which case z0 must be passed during
                forward()).
        """
        if isinstance(func, ContinuousRNNConverter):  # ugly hack
            hidden_channels = hidden_channels + input_channels

        super(NeuralCDE_forecasting, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        
        self.func = func
        self.initial = initial
        if initial and not isinstance(func, ContinuousRNNConverter):  # very ugly hack
            self.initial_network = torch.nn.Linear(input_channels, hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, input_channels-1)

    def extra_repr(self):
        return "input_channels={}, hidden_channels={}, output_channels={}, initial={}" \
               "".format(self.input_channels, self.hidden_channels, self.output_channels, self.initial)

    def forward(self, times, coeffs, final_index, z0=None, stream=False, **kwargs):
        """
        Arguments:
            times: The times of the observations for the input path X, e.g. as passed as an argument to
                `controldiffeq.natural_cubic_spline_coeffs`.
            coeffs: The coefficients describing the input path X, e.g. as returned by
                `controldiffeq.natural_cubic_spline_coeffs`.
            final_index: Each batch element may have a different final time. This defines the index within the tensor
                `times` of where the final time for each batch element is.
            z0: See the 'initial' argument to __init__.
            stream: Whether to return the result of the Neural CDE model at all times (True), or just the final time
                (False). Defaults to just the final time. The `final_index` argument is ignored if stream is True.
            **kwargs: Will be passed to cdeint.

        Returns:
            If stream is False, then this will return the terminal time z_T. If stream is True, then this will return
            all intermediate times z_t, for those t for which there was data.
        """

        # Extract the sizes of the batch dimensions from the coefficients

        coeff, _, _, _ = coeffs
        # in UEA : coeff : 32,181,4
        # in socar : coeff: 32,1205,11
        # in sepsis : coeffs : 1024,71,69
        batch_dims = coeff.shape[:-2] # 32
        if not stream:
            assert batch_dims == final_index.shape, "coeff.shape[:-2] must be the same as final_index.shape. " \
                                                    "coeff.shape[:-2]={}, final_index.shape={}" \
                                                    "".format(batch_dims, final_index.shape)

        cubic_spline = controldiffeq.NaturalCubicSpline(times, coeffs) # interpolated values
        # TODO z0
        if z0 is None:
            assert self.initial, "Was not expecting to be given no value of z0."
            if isinstance(self.func, ContinuousRNNConverter):  # still an ugly hack
                z0 = torch.zeros(*batch_dims, self.hidden_channels, dtype=coeff.dtype, device=coeff.device) # 32,32
            else:
                # 32,32,4
                XX = cubic_spline.evaluate(times[0]).float() # 32,4
                z0 = self.initial_network(XX) # aa0 32,32
        else:
            assert not self.initial, "Was expecting to be given a value of z0."
            if isinstance(self.func, ContinuousRNNConverter):  # continuing adventures in ugly hacks
                z0_extra = torch.zeros(*batch_dims, self.input_channels, dtype=z0.dtype, device=z0.device)
                z0 = torch.cat([z0_extra, z0], dim=-1)
            else:
                self.initial_network = None

        # for sepsis : z0 already exist
        # Figure out what times we need to solve for
        if stream:
            t = times
        else:
            # faff around to make sure that we're outputting at all the times we need for final_index.
            # in UEA : final_index : 32
            # in socar : final_index : 32
            # sorted_final_index : 뒤죽박죽인 final index sort,
            # inverse_final_index : sorted_final_index기준으로 final_index의 index 가지고 있음.
            sorted_final_index, inverse_final_index = final_index.unique(sorted=True, return_inverse=True)
            if 0 in sorted_final_index:
                sorted_final_index = sorted_final_index[1:]
                final_index = inverse_final_index
            else:
                final_index = inverse_final_index + 1
            if len(times) - 1 in sorted_final_index:
                sorted_final_index = sorted_final_index[:-1]

            t = torch.cat([times[0].unsqueeze(0), times[sorted_final_index], times[-1].unsqueeze(0)])

        # Switch default solver
        if 'method' not in kwargs:
            kwargs['method'] = 'rk4'
        if kwargs['method'] == 'rk4':
            if 'options' not in kwargs:
                kwargs['options'] = {}
            options = kwargs['options']
            if 'step_size' not in options and 'grid_constructor' not in options:
                time_diffs = times[1:] - times[:-1]
                options['step_size'] = time_diffs.min().item()

        # Actually solve the CDE

        # TODO
        # UEA. DATASET z0 : 32,32
        # SOCAR DATASET z0 : 32,32
        # SEPSIS DATASET z0 :1024,49

        z_t = controldiffeq.cdeint(dX_dt=cubic_spline.derivative,
                                   z0=z0,
                                   func=self.func,
                                   t=t,
                                   **kwargs)

        
        if stream:
            # z_t is a tensor of shape (times, ..., channels), so change this to (..., times, channels)
            for i in range(len(z_t.shape) - 2, 0, -1):
                z_t = z_t.transpose(0, i)
        else:
            # final_index is a tensor of shape (...)
            # z_t is a tensor of shape (times, ..., channels)
            final_index_indices = final_index.unsqueeze(-1).expand(z_t.shape[1:]).unsqueeze(0)
            z_t = z_t.gather(dim=0, index=final_index_indices).squeeze(0)

        # Linear map and return
        pred_y = self.linear(z_t[:,times.shape[0]-1:,:])


        return pred_y

class ANCDE_forecasting(torch.nn.Module):
    
    def __init__(self, func,func_g, input_channels,output_time, hidden_channels, output_channels,attention_channel,slope_check,soft, timewise,file,initial=True):
        
        if isinstance(func, ContinuousRNNConverter):  # ugly hack
            hidden_channels = hidden_channels + input_channels

        super(ANCDE_forecasting, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.output_time = output_time
        self.func_f = func 
        self.func_g = func_g 
        self.initial = initial
        self.attention_channel = attention_channel
        self.slope_check = slope_check
        self.soft = soft
        self.file =file
        self.STE = Hardsigmoid()
        self.binarizer = RoundST 
        if initial and not isinstance(func, ContinuousRNNConverter):  # very ugly hack
            self.initial_network = torch.nn.Linear(input_channels, input_channels)

        self.feature_extractor = torch.nn.Linear(input_channels,hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, input_channels) # hidden state -> prediction
        self.time_attention = torch.nn.Linear(input_channels,1)
        self.timewise = timewise
    def extra_repr(self):
        return "input_channels={}, hidden_channels={}, output_channels={}, initial={}" \
               "".format(self.input_channels, self.hidden_channels, self.output_channels, self.initial)

    def forward(self, times, coeffs, final_index,slope, z0=None, stream=False, **kwargs):
        

        # Extract the sizes of the batch dimensions from the coefficients
        
        coeff, _, _, _ = coeffs
        # SEPSIS : 1024,71,69
        batch_dims = coeff.shape[:-2] # 32
        if not stream:
            assert batch_dims == final_index.shape, "coeff.shape[:-2] must be the same as final_index.shape. " \
                                                    "coeff.shape[:-2]={}, final_index.shape={}" \
                                                    "".format(batch_dims, final_index.shape)
        
        cubic_spline = controldiffeq.NaturalCubicSpline(times, coeffs) # interpolated values
        
        if z0 is None:
            assert self.initial, "Was not expecting to be given no value of z0."
            if isinstance(self.func_f, ContinuousRNNConverter):  # still an ugly hack
                z0 = torch.zeros(*batch_dims, self.hidden_channels, dtype=coeff.dtype, device=coeff.device) # 32,32
            else:
                # z0 = self.initial_network(cubic_spline.evaluate(times[0]))  # 32,32,4
                XX = cubic_spline.evaluate(times[0]).float() # 32,4
                
                z0 = self.initial_network(XX) # aa0 32,32

        else:
            assert not self.initial, "Was expecting to be given a value of z0."
            if isinstance(self.func_f, ContinuousRNNConverter):  # continuing adventures in ugly hacks

                z0_extra = torch.zeros(*batch_dims, self.input_channels, dtype=z0.dtype, device=z0.device)
                z0 = torch.cat([z0_extra, z0], dim=-1)
            else:
                self.initial_network = None

        # Figure out what times we need to solve for
        if stream:
            t = times
        else:

            sorted_final_index, inverse_final_index = final_index.unique(sorted=True, return_inverse=True)
            if 0 in sorted_final_index:
                sorted_final_index = sorted_final_index[1:]
                final_index = inverse_final_index
            else:
                final_index = inverse_final_index + 1
            if len(times) - 1 in sorted_final_index:
                sorted_final_index = sorted_final_index[:-1]

            t = torch.cat([times[0].unsqueeze(0), times[sorted_final_index], times[-1].unsqueeze(0)])

        # Switch default solver
        if 'method' not in kwargs:
            kwargs['method'] = 'rk4'
        if kwargs['method'] == 'rk4':
            if 'options' not in kwargs:
                kwargs['options'] = {}
            options = kwargs['options']
            if 'step_size' not in options and 'grid_constructor' not in options:
                time_diffs = times[1:] - times[:-1]
                options['step_size'] = time_diffs.min().item()


        sigmoid = torch.nn.Sigmoid()
        # import pdb ; pdb.set_trace()
        self.atten_in = self.hidden_channels
        # import pdb ; pdb.set_trace()
        # attention = controldiffeq.cdeint(dX_dt=cubic_spline.derivative,
        #                            z0=z0,
        #                            func=self.func_f,
        #                            t=times,
        #                            **kwargs)
        attention = controldiffeq.ancde_bottom(dX_dt=cubic_spline.derivative,
                                   z0=z0,
                                   func=self.func_f,
                                   t=times,
                                   file=self.file,
                                   **kwargs)
        # import pdb ; pdb.set_trace()
        h_prime = np.load(self.file)
        # print(h_prime.shape)
        if self.timewise:
            
            attention = self.time_attention(attention)
            
        
        if self.soft :
            
            attention = sigmoid(attention)
        else:
            if self.slope_check :
                
                attention = self.STE(slope * attention)
                attention = self.binarizer(attention)
            else :
                
                attention = sigmoid(attention) # sigmoid -> 다른 것 
                attention = self.binarizer(attention)
        
        x0 = cubic_spline.evaluate(times[0]).float()
        a0 =  attention[0,:,:]
        y0 = torch.mul(x0,a0)
        y0 = self.feature_extractor(y0) 
        
        z_t = controldiffeq.ancde(dX_dt=cubic_spline.derivative,
                                   attention=attention,
                                   z0 =y0,
                                   X_s=cubic_spline,
                                   func_g = self.func_g,
                                   h_prime = h_prime,
                                   t=t,
                                   timewise=self.timewise,
                                   **kwargs)
        # import pdb ; pdb.set_trace()
        if stream:
            # z_t is a tensor of shape (times, ..., channels), so change this to (..., times, channels)
            for i in range(len(z_t.shape) - 2, 0, -1):
                z_t = z_t.transpose(0, i)
        else:
            # final_index is a tensor of shape (...)
            # z_t is a tensor of shape (times, ..., channels)
            final_index_indices = final_index.unsqueeze(-1).expand(z_t.shape[1:]).unsqueeze(0)
            z_t = z_t.gather(dim=0, index=final_index_indices).squeeze(0)

        # Linear map and return
        input_time = z_t.shape[1]
        # pred_y = self.linear(z_t[:,times.shape[0]-1:,:])
        
        pred_y = self.linear(z_t[:,input_time-self.output_time:,:])

        return pred_y

