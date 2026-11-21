"""Source package for the Sentiment Analysis System.

Modules
-------
config        : Project paths and shared constants.
nltk_setup    : Checks / downloads the small NLTK data packages we need.
preprocessing : Text cleaning, tokenisation, stop-word removal and stemming.
evaluation    : Metric calculation and chart generation.
train         : End-to-end training and model selection script.
predict       : Loads saved artifacts and predicts sentiment for new text.
"""

__version__ = "1.0.0"
