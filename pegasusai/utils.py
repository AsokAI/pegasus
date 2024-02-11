import inspect
from typing import Optional


def generate_pure_functions_from_class(
    class_type,
    object_name: str,
    function_prefix: Optional[str] = None,
    ignore_protected_functions: bool = False,
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
                    if ":" in str(param_type):
                        function_signature_str += (
                            str(param_type) + ", "
                            if param_type.annotation != inspect._empty
                            else param_name + ", "
                        )
                    else:
                        function_signature_str += (
                            param_name + ": " + str(param_type) + ", "
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
            mapped_function_name = function_prefix + name if function_prefix else name
            mapped_code += "def " + mapped_function_name + function_signature_str
            mapped_code += (
                " -> None:\n\t"
                if function_signature.return_annotation == inspect._empty
                else ":\n\t"
            )
            param_names = list(function_signature.parameters.keys())
            code_inside_function = ""
            for param_name in param_names:
                if param_name != "self":
                    code_inside_function += f"{param_name}={param_name}, \t"
            code_inside_function = (
                f"return {object_name}.{name}(" + code_inside_function + ")\n\n\n"
            )
            mapped_code += code_inside_function
    mapped_code = mapped_code.replace("NoneType", "None")
    mapped_code = mapped_code.replace(" kwargs,", " **kwargs,")
    return mapped_code
