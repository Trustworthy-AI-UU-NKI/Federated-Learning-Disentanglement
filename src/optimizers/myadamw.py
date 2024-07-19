from typing import Dict

import torch
from flwr.common.typing import NDArrays
from torch.optim import AdamW
from typing import List, Optional
import math
from torch import Tensor

class MyAdamW(AdamW):
    def __init__(self, params, ratio, **kwargs):
        super().__init__(params, **kwargs)

        self.ratio = ratio
        self.local_normalizing_vec = 0
        self.p_factor_total = 0
        self.local_counter = 0
        
    @torch.no_grad()
    def step(self, p_factor=None, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            amsgrad = group['amsgrad']
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                grads.append(p.grad)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = torch.zeros((1,), dtype=torch.float, device=p.device) \
                        if self.defaults['capturable'] else torch.tensor(0.)
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])

                if amsgrad:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                state_steps.append(state['step'])

            self.adamw(params_with_grad,
                  grads,
                  exp_avgs,
                  exp_avg_sqs,
                  max_exp_avg_sqs,
                  state_steps,
                  amsgrad=amsgrad,
                  beta1=beta1,
                  beta2=beta2,
                  lr=group['lr'],
                  weight_decay=group['weight_decay'],
                  eps=group['eps'],
                  maximize=group['maximize'],
                  foreach=group['foreach'],
                  capturable=group['capturable'],
                  p_factor=p_factor)
            
        # if self.momentum != 0:
        #     self.local_counter = self.local_counter * self.momentum + 1
        #     self.local_normalizing_vec += self.local_counter

        # if self.momentum == 0
        self.local_normalizing_vec += 1

        # if p_factor is not None:
        #         self.p_factor_total += p_factor
        # else:
        #     self.p_factor_total += 1

        return loss
    
    def get_gradient_scaling(self) -> Dict[str, float]:
        """Compute the scaling factor for local client gradients.

        Returns: A dictionary containing weight, tau, and local_norm.
        """
     
        local_tau = torch.tensor(self.local_normalizing_vec * self.ratio)
        local_stats = {
            # "weight": self.ratio,
            "tau": local_tau.item(),
            "local_norm": self.local_normalizing_vec,
            "p_factor_total": self.p_factor_total
        }

        return local_stats
    
    def adamw(self, params: List[Tensor],
            grads: List[Tensor],
            exp_avgs: List[Tensor],
            exp_avg_sqs: List[Tensor],
            max_exp_avg_sqs: List[Tensor],
            state_steps: List[Tensor],
            # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
            # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
            foreach: bool = None,
            capturable: bool = False,
            *,
            amsgrad: bool,
            beta1: float,
            beta2: float,
            lr: float,
            weight_decay: float,
            eps: float,
            maximize: bool,
            p_factor: float = None):
        r"""Functional API that performs AdamW algorithm computation.

        See :class:`~torch.optim.AdamW` for details.
        """

        if not all(isinstance(t, torch.Tensor) for t in state_steps):
            raise RuntimeError("API has changed, `state_steps` argument must contain a list of singleton tensors")

        if foreach is None:
            # Placeholder for more complex foreach logic to be added when value is not set
            foreach = False

        if foreach and torch.jit.is_scripting():
            raise RuntimeError('torch.jit.script not supported with foreach optimizers')

        if foreach and not torch.jit.is_scripting():
            func = self._multi_tensor_adamw
        else:
            func = self._single_tensor_adamw

        func(params,
            grads,
            exp_avgs,
            exp_avg_sqs,
            max_exp_avg_sqs,
            state_steps,
            amsgrad=amsgrad,
            beta1=beta1,
            beta2=beta2,
            lr=lr,
            weight_decay=weight_decay,
            eps=eps,
            maximize=maximize,
            capturable=capturable,
            p_factor=p_factor)


    def _single_tensor_adamw(self, params: List[Tensor],
                            grads: List[Tensor],
                            exp_avgs: List[Tensor],
                            exp_avg_sqs: List[Tensor],
                            max_exp_avg_sqs: List[Tensor],
                            state_steps: List[Tensor],
                            *,
                            amsgrad: bool,
                            beta1: float,
                            beta2: float,
                            lr: float,
                            weight_decay: float,
                            eps: float,
                            maximize: bool,
                            capturable: bool,
                            p_factor: float):
        
        for i, param in enumerate(params):
            grad = grads[i] if not maximize else -grads[i]
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            step_t = state_steps[i]

            if capturable:
                assert param.is_cuda and step_t.is_cuda, "If capturable=True, params and state_steps must be CUDA tensors."

            if torch.is_complex(param):
                grad = torch.view_as_real(grad)
                exp_avg = torch.view_as_real(exp_avg)
                exp_avg_sq = torch.view_as_real(exp_avg_sq)
                param = torch.view_as_real(param)

            # update step
            step_t += 1

            # Perform stepweight decay
            param.mul_(1 - lr * weight_decay)

            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            if capturable:
                step = step_t

                # 1 - beta1 ** step can't be captured in a CUDA graph, even if step is a CUDA tensor
                # (incurs "RuntimeError: CUDA error: operation not permitted when stream is capturing")
                bias_correction1 = 1 - torch.pow(beta1, step)
                bias_correction2 = 1 - torch.pow(beta2, step)

                step_size = lr / bias_correction1
                step_size_neg = step_size.neg()

                bias_correction2_sqrt = bias_correction2.sqrt()

                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                    # Uses the max. for normalizing running avg. of gradient
                    # Folds in (admittedly ugly) 1-elem step_size math here to avoid extra param-set-sized read+write
                    # (can't fold it into addcdiv_ below because addcdiv_ requires value is a Number, not a Tensor)
                    denom = (max_exp_avg_sqs[i].sqrt() / (bias_correction2_sqrt * step_size_neg)).add_(eps / step_size_neg)
                else:
                    denom = (exp_avg_sq.sqrt() / (bias_correction2_sqrt * step_size_neg)).add_(eps / step_size_neg)

                param.addcdiv_(exp_avg, denom)
            else:
                step = step_t.item()

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                step_size = lr / bias_correction1

                bias_correction2_sqrt = math.sqrt(bias_correction2)

                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sqs[i].sqrt() / bias_correction2_sqrt).add_(eps)
                else:
                    denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

                # update accumalated local updates
                param_state = self.state[param]
                cum_grad_step_size = step_size * p_factor if p_factor is not None else step_size
                
                if "cum_grad" not in param_state:
                    param_state["cum_grad"] = torch.zeros_like(grad)
                    param_state["cum_grad"].addcdiv_(exp_avg, denom, value=cum_grad_step_size)

                else:
                    param_state["cum_grad"].addcdiv_(exp_avg, denom, value=cum_grad_step_size)

                param.addcdiv_(exp_avg, denom, value=-step_size)


    def _multi_tensor_adamw(self, params: List[Tensor],
                            grads: List[Tensor],
                            exp_avgs: List[Tensor],
                            exp_avg_sqs: List[Tensor],
                            max_exp_avg_sqs: List[Tensor],
                            state_steps: List[Tensor],
                            *,
                            amsgrad: bool,
                            beta1: float,
                            beta2: float,
                            lr: float,
                            weight_decay: float,
                            eps: float,
                            maximize: bool,
                            capturable: bool):
        if len(params) == 0:
            return

        if capturable:
            assert all(p.is_cuda and step.is_cuda for p, step in zip(params, state_steps)), \
                "If capturable=True, params and state_steps must be CUDA tensors."

        if maximize:
            grads = torch._foreach_neg(tuple(grads))  # type: ignore[assignment]

        grads = [torch.view_as_real(x) if torch.is_complex(x) else x for x in grads]
        exp_avgs = [torch.view_as_real(x) if torch.is_complex(x) else x for x in exp_avgs]
        exp_avg_sqs = [torch.view_as_real(x) if torch.is_complex(x) else x for x in exp_avg_sqs]
        params = [torch.view_as_real(x) if torch.is_complex(x) else x for x in params]

        # update steps
        torch._foreach_add_(state_steps, 1)

        # Perform stepweight decay
        torch._foreach_mul_(params, 1 - lr * weight_decay)

        # Decay the first and second moment running average coefficient
        torch._foreach_mul_(exp_avgs, beta1)
        torch._foreach_add_(exp_avgs, grads, alpha=1 - beta1)

        torch._foreach_mul_(exp_avg_sqs, beta2)
        torch._foreach_addcmul_(exp_avg_sqs, grads, grads, 1 - beta2)

        if capturable:
            # TODO: use foreach_pow if/when foreach_pow is added
            bias_correction1 = [torch.pow(beta1, step) for step in state_steps]
            bias_correction2 = [torch.pow(beta2, step) for step in state_steps]
            # foreach_sub doesn't allow a scalar as the first arg
            torch._foreach_sub_(bias_correction1, 1)
            torch._foreach_sub_(bias_correction2, 1)
            torch._foreach_neg_(bias_correction1)
            torch._foreach_neg_(bias_correction2)

            # foreach_div doesn't allow a scalar as the first arg
            step_size = torch._foreach_div(bias_correction1, lr)
            torch._foreach_reciprocal_(step_size)
            torch._foreach_neg_(step_size)

            bias_correction2_sqrt = torch._foreach_sqrt(bias_correction2)

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch._foreach_maximum_(max_exp_avg_sqs, exp_avg_sqs)

                # Use the max. for normalizing running avg. of gradient
                max_exp_avg_sq_sqrt = torch._foreach_sqrt(max_exp_avg_sqs)
                # Folds in (admittedly ugly) 1-elem step_size math here to avoid extra param-set-sized read+write
                # (can't fold it into addcdiv_ below because addcdiv_ requires value is a Number, not a Tensor)
                torch._foreach_div_(max_exp_avg_sq_sqrt, torch._foreach_mul(bias_correction2_sqrt, step_size))
                eps_over_step_size = torch._foreach_div(step_size, eps)
                torch._foreach_reciprocal_(eps_over_step_size)
                denom = torch._foreach_add(max_exp_avg_sq_sqrt, eps_over_step_size)
            else:
                exp_avg_sq_sqrt = torch._foreach_sqrt(exp_avg_sqs)
                torch._foreach_div_(exp_avg_sq_sqrt, torch._foreach_mul(bias_correction2_sqrt, step_size))
                eps_over_step_size = torch._foreach_div(step_size, eps)
                torch._foreach_reciprocal_(eps_over_step_size)
                denom = torch._foreach_add(exp_avg_sq_sqrt, eps_over_step_size)

            torch._foreach_addcdiv_(params, exp_avgs, denom)
        else:
            bias_correction1 = [1 - beta1 ** step.item() for step in state_steps]
            bias_correction2 = [1 - beta2 ** step.item() for step in state_steps]

            step_size = [(lr / bc) * -1 for bc in bias_correction1]

            bias_correction2_sqrt = [math.sqrt(bc) for bc in bias_correction2]

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch._foreach_maximum_(max_exp_avg_sqs, exp_avg_sqs)

                # Use the max. for normalizing running avg. of gradient
                max_exp_avg_sq_sqrt = torch._foreach_sqrt(max_exp_avg_sqs)
                torch._foreach_div_(max_exp_avg_sq_sqrt, bias_correction2_sqrt)
                denom = torch._foreach_add(max_exp_avg_sq_sqrt, eps)
            else:
                exp_avg_sq_sqrt = torch._foreach_sqrt(exp_avg_sqs)
                torch._foreach_div_(exp_avg_sq_sqrt, bias_correction2_sqrt)
                denom = torch._foreach_add(exp_avg_sq_sqrt, eps)

