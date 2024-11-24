import sys
from logging import getLogger

from config import Config
from data.dataloader import construct_dataloader
from data.dataset import create_datasets
from generate.data import generate_sample_data
from generate.model import generate_sample_model
from trainer import Trainer
from utils import init_seed, init_logger, dynamic_load


def get_arguments():
    args = dict()
    for arg in sys.argv[1:]:
        arg_name, arg_value = arg.split('=')
        try:
            arg_value = int(arg_value)
        except:
            try:
                arg_value = float(arg_value)
            except:
                pass
        arg_name = arg_name.strip('-')
        args[arg_name] = arg_value
    print(args)
    return args

def main_process(model='DPGNN', config_dict=None, saved=True):
    """Main process API for experiments of VPJF

    Args:
        model (str): Model name.
        config_dict (dict): Parameters dictionary used to modify experiment parameters.
            Defaults to ``None``.
        saved (bool): Whether to save the model parameters. Defaults to ``True``.
    """

    # configurations initialization
    config = Config(model, config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)
    generate_sample_data(config['dataset_path'])
    generate_sample_model(config['dataset_path'])

    # data preparation
    pool = dynamic_load(config, 'data.pool', 'Pool')(config)
    logger.info(pool)

    datasets = create_datasets(config, pool)
    for ds in datasets:
        logger.info(ds)

    # load dataset
    train_data, valid_data_g, valid_data_j, test_data_g, test_data_j = construct_dataloader(config, datasets)

    # model loading and initialization
    model = dynamic_load(config, 'model')(config, pool).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = Trainer(config, model)

    # model training
    best_valid_score, best_valid_result_g, best_valid_result_j = trainer.fit(train_data, valid_data_g, valid_data_j, saved=saved)

    logger.info('best valid result for geek: {}'.format(best_valid_result_g))
    logger.info('best valid result for job: {}'.format(best_valid_result_j))

    # model evaluation for user
    test_result, test_result_str = trainer.evaluate(test_data_g, load_best_model=True)
    logger.info('test for user result [all]: {}'.format(test_result_str))

    # model evaluation for job
    test_result, test_result_str = trainer.evaluate(test_data_j, load_best_model=True, reverse=True)
    logger.info('test for job result [all]: {}'.format(test_result_str))

    return {
        'best_valid_score': best_valid_score,
        'best_valid_result_g': best_valid_result_g,
        'best_valid_result_j': best_valid_result_j,
        'test_result': test_result
    }


if __name__ == "__main__":
    args = get_arguments()
    main_process(model='DPGNN', config_dict=args)
