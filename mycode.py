import os


from glue_evaluator import GLUEvaluator, set_seed



evaluator = GLUEvaluator.load('model/distillBERT1', 0)

evaluator.plot_terms_changes(os.path.join('output', 'bias_term_changes'))