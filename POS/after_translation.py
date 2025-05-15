from src.prepare_for_training import PrepareForTraining
from src.utils import create_config, create_commandline_args

args = create_commandline_args()
config = create_config(args)

preparation_it = PrepareForTraining(config, args.language).main()