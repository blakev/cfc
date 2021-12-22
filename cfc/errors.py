#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# >>
#   cfc, 2021
#   Blake VandeMerwe <blake@vidangel.com>
# <<

from click import UsageError


__all__ = [
    'ConfigurationError',
]


class ConfigurationError(UsageError):
    """Exceptions raised when parsing and merging configuration files."""


class RuleError(UsageError):
    """Exceptions raised while parsing or evaluating Path Rules."""
