import os


from glue_evaluator import GLUEvaluator, set_seed
PADDING = "max_length"
MAX_SEQUENCE_LEN = 128



def get_test_accuracy(evaluator):
    evaluator.plot_terms_changes(os.path.join('tmp', 'bias_term_changes'))

    

    # evaluator.preprocess_dataset(PADDING, MAX_SEQUENCE_LEN, 8)
    # evaluator.export_model_test_set_predictions('tmp')

    # if evaluator.device is not None:
    #     evaluator.model.cuda(evaluator.device)

    # for dataloader_type, dataloader in evaluator.data_loaders.items():
    #     if str(dataloader_type) == 'test':
    #         results = evaluator._evaluate(dataloader, dataloader_type.upper())
    #         for metric_name, result in results.items():
    #             print(metric_name, result)

evaluator = GLUEvaluator.load('output\distilbert-base-uncased_mrpc_bitfit_all_16_0.001\evaluator', 0)

get_test_accuracy(evaluator)

