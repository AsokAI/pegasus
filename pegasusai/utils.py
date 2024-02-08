import inspect
from typing import Optional

from datasets import DatasetDict


def generate_map_functions_from_class(
    class_type,
    object_name: str,
    ignore_protected_functions: bool = True,
    function_name: Optional[str] = None,
) -> str:
    mapped_code = ""
    for name, obj in inspect.getmembers(class_type):
        if inspect.isfunction(obj) and (
            True if function_name is None else name == function_name
        ):
            if ignore_protected_functions and name.startswith("_"):
                continue
            function_signature = inspect.signature(obj)
            function_signature_str = function_signature.__str__()
            if "self" in function_signature_str:
                function_signature_str = function_signature_str.replace(
                    "self", f"{object_name}: {class_type.__name__}"
                )
            else:
                function_signature_str = ""
                param_names = list(function_signature.parameters.keys())
                param_types = list(function_signature.parameters.values())
                for param_name, param_type in zip(param_names, param_types):
                    function_signature_str += (
                        param_name + ": " + param_type.annotation.__name__ + ", "
                        if param_type.annotation != inspect._empty
                        else param_name + ", "
                    )
                function_signature_str = (
                    "("
                    + object_name
                    + ": "
                    + class_type.__name__
                    + ", "
                    + function_signature_str
                    + ")"
                )
            mapped_code += "def " + name + function_signature_str
            mapped_code += (
                " -> None:\n\t"
                if function_signature.return_annotation == inspect._empty
                else ":\n\t"
            )
            param_names = list(function_signature.parameters.keys())
            code_inside_function = (
                f"return {object_name}.{name}(" + ", ".join(param_names) + ")\n\n\n"
            )
            code_inside_function = code_inside_function.replace("self", object_name)
            mapped_code += code_inside_function
    mapped_code = mapped_code.replace("NoneType", "None")
    return mapped_code
