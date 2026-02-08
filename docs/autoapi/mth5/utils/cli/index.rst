mth5.utils.cli
==============

.. py:module:: mth5.utils.cli

.. autoapi-nested-parse::

   MTH5 CLI Tools
   ==============

   Command-line interface tools for MTH5 file operations.

   Created on February 7, 2026

   :copyright: MTH5 Development Team
   :license: MIT



Functions
---------

.. autoapisummary::

   mth5.utils.cli.validate_command
   mth5.utils.cli.main


Module Contents
---------------

.. py:function:: validate_command(args: argparse.Namespace) -> int

   Execute validation command.

   :param args: Parsed command-line arguments.
   :type args: argparse.Namespace

   :returns: Exit code (0 for success, 1 for failure).
   :rtype: int


.. py:function:: main() -> int

   Main entry point for mth5-cli tool.

   :returns: Exit code.
   :rtype: int


