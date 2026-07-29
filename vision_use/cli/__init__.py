"""Command line interface.

The CLI is a thin presentation layer over the framework. It parses arguments,
configures logging, calls into the library, and renders results. It contains no
media logic, so that anything reachable from the terminal is equally reachable
from the Python API and from the MCP server that follows in M5.
"""
