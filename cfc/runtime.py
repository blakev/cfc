#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# >>
#   cfc, 2021
#   Blake VandeMerwe <blake@vidangel.com>
# <<

import glob
import random
import asyncio
from asyncio import AbstractEventLoop
from pathlib import Path
from functools import partial
from typing import List, Callable, Optional

from toolz.functoolz import curry

from cfc.utils import OutFnT, now, identity
from cfc.models import (
    Run,
    RunResult,
    DelHandlerT,
    CleanHandlerT,
    Configuration,
)


def default_del_handler(
    out: OutFnT,
    err: OutFnT,
    dry_run: bool,
    file: Path,
) -> bool:
    """(Curried) Attempt to delete a PosixPath, files only."""
    if file.exists() and file.is_file():
        try:
            if not dry_run:
                file.unlink()
        except IOError:
            error = True
        else:
            error = False
    else:
        error = True

    if dry_run:
        verb = '(dry)'
    else:
        verb = ''

    if error:
        err(f'..{verb} error deleting {file}')
    else:
        out(f'..{verb} deleted {file}')
    return not error


async def default_handler(
    out: OutFnT,
    err: OutFnT,
    del_fn: DelHandlerT,
    conf: Configuration,
) -> RunResult:
    """Default handler."""

    total_bytes = 0
    deleted: List[Path] = []

    for path in conf.active_paths:
        for file in glob.iglob(f'{path}/**', recursive=conf.recursive):
            file = Path(file)

            if not file.is_file():
                continue

            if conf.skip_extensions:
                if file.suffix in conf.skip_extensions:
                    continue

            if conf.include_extensions:
                if file.suffix not in conf.include_extensions:
                    continue

            for rule in conf.rules:
                if not rule.evaluate(file):
                    break
            else:
                # check if there are other files in the same folder that share a
                #  file stem, e.g. file0.png file0.md5 file0.info
                if conf.group_file_stem:
                    files = glob.iglob(f'{file.parent}/{file.stem}*')
                    files = list(map(Path, files))
                else:
                    files = [file]

                # if we didn't break from a failed rule..
                for f in files:
                    if f in deleted:
                        continue
                    # track the number of bytes we've cleaned up
                    stat = f.stat()
                    pending_bytes = stat.st_size

                    if del_fn(f):
                        total_bytes += pending_bytes
                        deleted.append(f)
    # ~~
    return RunResult(deleted, total_bytes)


async def forever(
    configs: List[Configuration],
    s_out: OutFnT,
    s_err: OutFnT,
    dry_run: bool = False,
    handler: Optional[CleanHandlerT] = None,
    delete_handler: Callable[[Path], bool] = None,
) -> None:
    """Main event loop for processing notifications and scheduled scans."""

    loop = asyncio.get_running_loop()

    if delete_handler is None:
        # our default delete handler can handle dry_run, so we can use it to unlink
        #  individual files with a little more pizzazz
        delete_handler = curry(default_del_handler, s_out, s_err, dry_run)
    else:
        # CATCH ALL, if the user is supplying a delete_handler then we can't
        #  assume it will correctly handle a dry_run=True. So, we substitute their
        # handler with a simple "print" to console.
        if dry_run:
            delete_handler = s_out

    if handler is None:
        # default handler with stdout/stderr channels
        handler = default_handler
    # pre-bind our functions for processing files
    handler = curry(handler, s_out, s_err, delete_handler)

    def print_result(t: asyncio.Task) -> None:
        res: Run = t.result()
        s_out(res.to_string())

    def every_fn(conf: Configuration) -> float:
        return conf.every.total_seconds()

    # get the lowest interval from all of our configurations
    #  and use it as a base to compute the sleep interval between checking
    # which configurations to clean up
    lowest = sorted(configs, key=every_fn)[0].every.total_seconds()

    while True:
        for c in configs:
            # it's time to run this configuration again
            if c.can_run(now()):
                if c.last_run is None:
                    s_out(f'scheduling {c}')
                # noinspection PyTypeChecker
                task = loop.create_task(c.cleanup(handler))
                task.add_done_callback(print_result)

        # get the next available run
        sleep_for = max(1.0, random.uniform(lowest * 0.4, lowest * 0.5))
        await asyncio.sleep(sleep_for)
