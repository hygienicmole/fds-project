"""
Adversarial ML API Package

This package provides a FastAPI-based REST API for generating and analyzing
adversarial examples using FGSM and PGD attacks.

Modules:
    backend: Main FastAPI application with all endpoints
"""

__version__ = "1.0.0"
__author__ = "CS685 Adversarial ML Project"

from .backend import app

__all__ = ['app']
