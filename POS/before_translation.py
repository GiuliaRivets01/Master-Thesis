from src.data_loader import PrepareForTranslation
from src.utils import create_commandline_args, create_config
from src.config import load_config

args = create_commandline_args()
config = create_config(args)

PrepareForTranslation(config['dataset']['train_path_conllu'], 
                      config['dataset']['val_path_conllu'],
                      config['dataset']['test_path_conllu'], args.language).main()
