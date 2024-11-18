from collections import OrderedDict
from collections.abc import Iterator
from typing import Tuple

from tensor import Tensor


class Module:
    def __init__(self) -> None:
        self._parameters = OrderedDict()
        self._modules = OrderedDict()

    def forward(self, *inputs):
        raise NotImplementedError

    def __call__(self, *inputs):
        return self.forward(*inputs)

    def parameters(self, include_submodules: bool = True) -> Iterator['Tensor']:
        for name, param in self._parameters.items():
            if param is not None:
                yield param
        if include_submodules:
            for module in self._modules.values():
                if module is not None:
                    yield from module.parameters(include_submodules=True)

    def named_parameters(self, include_submodules: bool = True) -> Iterator[Tuple[str, 'Tensor']]:
        for name, param in self._parameters.items():
            if param is not None:
                yield name, param

        if include_submodules:
            for name, module in self._modules.items():
                if module is not None:
                    yield from module.named_parameters()

    def __setattr__(self, name: str, value):
        if isinstance(value, Module):
            self._modules[name] = value
        elif isinstance(value, Tensor):
            self._parameters[name] = value
        super().__setattr__(name, value)

    def __repr__(self) -> str:
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            child_lines.append(f'({key}): {mod_str}')
        lines = child_lines

        main_str = self.__class__.__name__ + '('
        if lines:
            main_str += '\n  ' + '\n  '.join(lines) + '\n'
        main_str += ')'
        return main_str
