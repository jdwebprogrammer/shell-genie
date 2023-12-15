import platform
from .backends import OpenAIGenie


def get_os_info():
    oper_sys = platform.system()
    if oper_sys == "Windows" or oper_sys == "Darwin":
        oper_sys = "MacOS" if oper_sys == "Darwin" else "Windows"
        return (oper_sys, platform.platform(aliased=True, terse=True))
    if oper_sys == "Linux":
        return (oper_sys, platform.freedesktop_os_release()["PRETTY_NAME"])
    return (None, None)


def get_backend(**config: dict):
    backend_name = config["backend"]
    return TrueOpenGenie(os_fullname=config["os_fullname"],shell=config["shell"])
    else:
        raise ValueError(f"Unknown backend: {backend_name}")
