#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# >>
#   cfc, 2021
#   Blake VandeMerwe <blake@vidangel.com>
# <<

import sys
import asyncio
from pathlib import Path
from typing import Tuple, Optional

import click
import setproctitle

from cfc.utils import get_output
from cfc.models import Configuration
from cfc.runtime import forever


@click.command()
@click.option(
    '-d',
    '--dry-run',
    default=False,
    type=bool,
    show_default=True,
    is_flag=True,
    help="Do not delete any files, display action instead.",
)
@click.option(
    '-s',
    '--silent',
    default=False,
    type=bool,
    show_default=True,
    is_flag=True,
    help="Do not output anything to console.",
)
@click.argument(
    'configs',
    type=click.Path(
        exists=True,
        dir_okay=True,
        resolve_path=True,
        path_type=Path,
    ),
    nargs=-1,
)
def main(
    dry_run: bool,
    silent: bool,
    configs: Tuple[Path, ...],
) -> int:
    """Cache folder cleaner (cfc)"""

    s_out, s_err = get_output(silent=silent)
    configs = Configuration.resolve_configs(*configs)

    # creat our new async context
    loop = asyncio.new_event_loop()
    err: Optional[Exception] = None

    try:
        loop.run_until_complete(forever(
            configs,
            s_out,
            s_err,
            dry_run=dry_run,
        ))

    except KeyboardInterrupt:
        pass

    except Exception as e:
        s_err(f'{e.__class__.__name__}: ({e.__traceback__.tb_lineno}) {e}')
        err = e
        raise e

    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
        return 1 if err else 0


# set the process name to distinguish it from other Python processes
setproctitle.setproctitle('cfc')
sys.exit(main())
