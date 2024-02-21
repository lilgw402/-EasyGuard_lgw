def _print_args_cli(args, meta="cli"):
    """Print arguments."""
    print(
        "------------------------ " + meta + " arguments ------------------------",
        flush=True,
    )
    str_list = []
    for arg in vars(args):
        dots = "." * (48 - len(arg))
        str_list.append("  {} {} {}".format(arg, dots, getattr(args, arg)))
    for arg in sorted(str_list, key=lambda x: x.lower()):
        print(arg, flush=True)


def print_cfg(cfg):
    if "data" in cfg:
        _print_args_cli(cfg["data"], "data")
    if "model" in cfg:
        _print_args_cli(cfg["model"], "model")
    if "trainer" in cfg:
        _print_args_cli(cfg["trainer"], "trainer")
    print(
        "-----------------------------------------------------------------\n",
        flush=True,
    )
