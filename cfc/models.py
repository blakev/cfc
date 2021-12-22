#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# >>
#   cfc, 2021
#   Blake VandeMerwe <blake@vidangel.com>
# <<
import os
import pwd
import json
import math
import asyncio
import operator as op
import mimetypes
from enum import Enum
from pathlib import Path
from functools import cached_property
from collections import deque
from typing import (
    List,
    Deque,
    Union,
    TypeVar,
    Callable,
    Optional,
    Awaitable,
)

import pytz as pytz
from humanize import naturalsize
from pendulum import Period, DateTime, Duration
from pydantic import (
    Extra,
    Field,
    BaseModel,
    PrivateAttr,
    ValidationError,
    validator,
)
from pydantic.dataclasses import dataclass

from cfc.utils import OutFnT, now, s_err
from cfc.errors import RuleError, ConfigurationError


__all__ = [
    'BinOpT',
    'CleanHandlerT',
    'Configuration',
    'DelHandlerT',
    'Rule',
    'Run',
    'RunResult',
]

T = TypeVar('T')
V = TypeVar('V')
BinOpT = Callable[[T, V], bool]


class Fns(Enum):
    """Comparison functions available for attributes."""

    EQ = '=='
    NE = '!='
    GT = '>'
    GE = '>='
    LT = '<'
    LE = '<='


def noop_fn(a, b):
    return False


op_funcs = {
    Fns.EQ: op.eq,
    Fns.NE: op.ne,
    Fns.GT: op.gt,
    Fns.GE: op.ge,
    Fns.LT: op.lt,
    Fns.LE: op.le,
    'noop': noop_fn
}


@dataclass
class ComputePath:
    path: Path

    @cached_property
    def stat(self) -> os.stat_result:
        return self.path.stat()

    @property
    def age_seconds(self) -> int:
        a = now()
        b = DateTime.fromtimestamp(self.stat.st_ctime, tz=pytz.utc)
        return math.floor((a - b).total_seconds())

    @property
    def age_hours(self) -> int:
        a = now()
        b = DateTime.fromtimestamp(self.stat.st_ctime, tz=pytz.utc)
        return math.floor((a - b).total_hours())

    @property
    def user(self) -> str:
        return pwd.getpwuid(self.stat.st_uid).pw_name

    @property
    def mimetype(self) -> Optional[str]:
        return mimetypes.guess_type(self.path)[0]


class Rule(BaseModel):
    # yapf: disable
    attr:   str
    fn:     BinOpT
    value:  Union[str, int, float, None]
    # yapf: enable

    @validator('fn', pre=True)
    def fix_fn(cls, v) -> BinOpT:
        return op_funcs.get(Fns(v), noop_fn)

    @validator('attr')
    def fix_attr(cls, v: str) -> str:
        if not v.isidentifier():
            ValidationError(v)
        return v

    @validator('value')
    def fix_value(cls, v: str) -> Union[str, int, float, None]:
        o_v = v

        if v.lower() == 'none':
            return None

        if v[0] == '-':
            v = v[1:]
            neg = -1
        else:
            neg = 1

        if v.isdecimal():
            return int(v) * neg

        if v.count('.') == 1 and all(map(str.isdecimal, v.split('.'))):
            return float(v) * float(neg)

        return o_v

    @property
    def is_valid(self) -> bool:
        return self.fn is not noop_fn

    @classmethod
    def from_string(cls, v: str) -> 'Rule':
        pieces = v.split()[:3]
        params = zip(['attr', 'fn', 'value'], pieces)
        obj = cls(**dict(params))
        return obj

    def evaluate(self, path: Path) -> bool:
        """Determines if a given file ``path`` meets the rule criteria."""

        computed = ComputePath(path)
        sources = (
            path,
            path.stat(),
            computed,
        )

        for source in sources:
            if hasattr(source, self.attr):
                a = getattr(source, self.attr)
                b = self.value
                return self.fn(a, b)
        # ~ we don't have this attribute, so it's an error
        raise RuleError(f'could not locate {self.attr=}')


@dataclass
class Run:
    """Represents a run of `cleanup`, to track when it happened and what files were
    removed/purged."""

    # yapf: disable
    started:        DateTime
    finished:       DateTime
    files:          List[Path]
    bytes_freed:    int
    # yapf: enable

    @property
    def empty(self) -> bool:
        return len(self.files) == 0

    @property
    def elapsed(self) -> Period:
        return self.finished - self.started

    def to_string(self) -> str:
        b = naturalsize(self.bytes_freed)
        s = self.elapsed.in_words()
        return f'removed {len(self.files)} totalling {b} took={s}'


@dataclass(frozen=True)
class RunResult:
    # yapf: disable
    files:          List[Path]
    bytes_freed:    int
    # yapf: enable


class Configuration(BaseModel):
    """Defines one or more scanning configurations for an instance of ``cfc``."""

    class Config:
        extra = Extra.ignore
        underscore_attrs_are_private = True

    # yapf: disable
    enabled:            bool = True
    group_file_stem:    bool = False
    recursive:          bool = True
    every:              Duration = Field(default=Duration(minutes=1))
    paths:              List[Path] = Field(default_factory=list)
    skip_extensions:    List[str] = Field(default_factory=list)
    include_extensions: List[str] = Field(default_factory=list)
    rules:              List[Rule] = Field(default_factory=list)
    # ~~ privates
    _runs:              Deque[Run] = PrivateAttr(default_factory=lambda: deque([], maxlen=30))
    _lock:              asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)
    # yapf: enable

    def __str__(self) -> str:
        every = self.every
        paths = ':'.join(map(str, self.paths))
        skip = ','.join(self.skip_extensions)
        incl = ','.join(self.include_extensions)
        return f'Configuration<every={every}, {paths}, {skip=} {incl=}>'

    @validator('every', pre=True)
    def fix_every(cls, v):
        if isinstance(v, Duration):
            return v
        d = Duration(**v)

        if d.total_seconds() <= 1.0:
            return Duration(seconds=1)
        # strip sub-second timing from our Duration
        return Duration(seconds=d.total_seconds())

    @validator('rules', pre=True, each_item=True)
    def fix_rules(cls, v) -> Rule:
        return Rule.from_string(v)

    @validator(
        'skip_extensions',
        'include_extensions',
        each_item=True,
        pre=True,
    )
    def fix_extensions(cls, v) -> List[str]:
        return '.' + v.lower().lstrip('.')

    @validator('paths', pre=True, each_item=True)
    def fix_paths(cls, v) -> Path:
        return Path(v)

    @property
    def last_run(self) -> Optional[DateTime]:
        """Returns the finish time of the last successful run."""
        if len(self._runs) == 0:
            return None
        return self._runs[-1].finished

    @property
    def next_run(self) -> DateTime:
        """Returns when the next run is allowed to execute."""
        if not self.last_run:
            return now()
        return self.last_run + self.every

    @property
    def active_paths(self) -> List[Path]:
        """Returns all the paths defined in the configuration that exist."""
        if not self.enabled:
            return []
        return [o for o in self.paths if o.exists() and o.is_dir()]

    def can_run(self, dt: DateTime = None) -> bool:
        """Determine if we can run this configuration."""
        if dt is None:
            dt = now()
        conditions = [
            self.enabled,
            self.last_run is None or self.next_run <= dt,
            self.active_paths,
            not self._lock.locked(),
        ]
        return all(conditions)

    async def cleanup(self, handler: 'CleanHandlerT') -> Run:
        async with self._lock:
            started = now()
            res = await handler(self)
            run = Run(
                started=started,
                finished=now(),
                files=res.files,
                bytes_freed=res.bytes_freed,
            )
            self._runs.append(run)
        return run

    @classmethod
    def resolve_configs(cls, *configs: Path) -> List['Configuration']:
        """Load one or more configurations and squash them together.

        Returns a single Config instance from all config paths passed as arguments. Only
        JSON-style configurations are supported at this time.

        If no valid configs are found the program will exit with an error.
        """
        pending = []

        for conf in configs:
            if conf.suffix.lower() not in ('.json',):
                s_err(f'invalid configuration {conf}')
                continue

            with open(conf.as_posix(), 'r') as fp:
                try:
                    loaded = Configuration.parse_obj(json.load(fp))
                except Exception as e:
                    s_err(e)
                    raise ConfigurationError('no valid configurations') from e
                else:
                    if loaded.enabled:
                        pending.append(loaded)

        if not pending:
            raise ConfigurationError('no valid configs')
        return pending


DelHandlerT = Callable[[Path], bool]
CleanHandlerT = Callable[[OutFnT, OutFnT, DelHandlerT, Configuration],
                         Awaitable[RunResult]]
