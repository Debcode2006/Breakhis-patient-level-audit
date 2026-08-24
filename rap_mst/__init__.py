"""RAP-MST: Retrieval-Augmented Pathology Memory Swin Transformer.

Research framework for patient-level binary classification (benign vs malignant)
on the BreaKHis dataset across all magnifications with a single unified model.

The package is organized so that future modules -- Retrieval Memory, Prototype
Learning, and a Reasoning Module -- can be added without refactoring the core
data / model / training abstractions. See the paper for the architecture
philosophy and roadmap.
"""

__version__ = "0.1.0"
