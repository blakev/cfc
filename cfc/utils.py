#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# >>
#   cfc, 2021
#   Blake VandeMerwe <blake@vidangel.com>
# <<

from functools import partial
from typing import Any, Tuple, Callable

import click
import pendulum


__all__ = [
    'OutFnT',
    's_err',
    's_out',
    'identity',
    'get_output',
    'now',
]

OutFnT = Callable[..., Any]

s_out = partial(click.secho, err=False, fg='green')
s_err = partial(click.secho, err=True, fg='red')
# aliases for printing colored output to the console


def identity(*_, **__) -> Any:
    return


def get_output(silent: bool = False) -> Tuple[OutFnT, OutFnT]:
    """Return our output functions, whether `silent` is activated via cli."""

    if silent:
        return identity, identity
    # ~~
    return s_out, s_err


def now() -> pendulum.DateTime:
    return pendulum.now('utc')
