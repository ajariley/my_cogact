from importlib import import_module


_EXPORTS = {
    "train_distillation": (".train", "train_distillation"),
    "load_teacher": (".loaders", "load_teacher"),
    "load_student": (".loaders", "load_student"),
    "load_dataloader": (".loaders", "load_dataloader"),
    "get_student_timesteps": (".runners", "get_student_timesteps"),
    "run_teacher_with_recording": (".runners", "run_teacher_with_recording"),
    "run_student_ddim_with_recording": (".runners", "run_student_ddim_with_recording"),
    "compute_loss": (".loss", "compute_loss"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value
