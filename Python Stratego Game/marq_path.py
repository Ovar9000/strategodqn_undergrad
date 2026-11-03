"""Puts Modular Stratego, the training code, on sys.path so the GUI can import its packages
(environment, network, settings). Imported by the GUI modules that use them."""
import os
import sys

MODULAR_STRATEGO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Modular Stratego"))
if MODULAR_STRATEGO not in sys.path:
    sys.path.append(MODULAR_STRATEGO)
