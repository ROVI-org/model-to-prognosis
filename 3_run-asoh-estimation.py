from concurrent.futures import ProcessPoolExecutor, as_completed
from argparse import ArgumentParser
from datetime import datetime
from copy import deepcopy
from pathlib import Path
import pickle as pkl
import logging
import sys

from moirae.interface import run_online_estimate
from moirae.interface.hdf5 import HDF5Writer

if __name__ == "__main__":
    # Parse user arguments
    parser = ArgumentParser()
    parser.add_argument('--data-directory', help='Directory containing data in HDF5 format')
    parser.add_argument('--starting-estimator', help='Path to a pickle file containing the initial state estimator')
    parser.add_argument('--config', default='serial', help='Configuration used to run the cells. "Serial" for single thread, "local" for all local cores. ')
    args = parser.parse_args()

    # Start the system
    logger = logging.getLogger('main')
    handlers = [logging.StreamHandler(sys.stdout)]
    for handler in handlers:
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        for my_logger in [logger]:
            my_logger.addHandler(handler)
            my_logger.setLevel(logging.INFO)

    # Load the PKL containing the state estimator
    estimator_path = Path(args.starting_estimator)
    with open(args.starting_estimator, 'rb') as fp:
        init_estimator = pkl.load(fp)
    logger.info(f'Loaded a {type(init_estimator)} from {args.starting_estimator}')

    # Create the output directory
    input_dir = Path(args.data_directory)
    output_dir = Path('estimates') / f'{input_dir.name}-{estimator_path.with_suffix("").name}-{datetime.now().strftime("%y%m%dT%H%M%S")}'
    file_count = len(list(input_dir.glob("*.hdf5")))
    logger.info(f'Reading {file_count} from {input_dir}. Writing outputs to {output_dir}')

    output_dir.mkdir(parents=True)
    with (output_dir / 'estimator.pkl').open('wb') as fp:
        pkl.dump(init_estimator, fp)

    # Make the parallel executor
    if args.config == 'serial':
        pool = ProcessPoolExecutor(1)
    elif args.config == 'local':
        pool = ProcessPoolExecutor()
    else:
        raise NotImplementedError(f'No such config: {args.config}')
    logger.info(f'Readied executor for config {args.config}: {pool}')

    # Loop over the example files
    with pool:
        futures = []
        for file in input_dir.glob("*.hdf5"):
            # Open up the writer
            cell_name = file.with_suffix('').name
            out_path = output_dir / file.with_suffix('.estimates.hdf5').name
            writer = HDF5Writer(hdf5_output=out_path, per_cycle='full', per_timestep='mean_cov', resizable=True)

            # Run the estimation
            my_estimator = deepcopy(init_estimator)
            future = pool.submit(run_online_estimate, file, pbar=args.config == 'serial', estimator=my_estimator, output_states=False, hdf5_output=writer)
            future.file = file
            futures.append(future)

        for i, future in enumerate(as_completed(futures)):
            file = future.file
            if (exc := future.exception()) is None:
                logger.warning(f'Finished running {file}. Failed with exception: {exc}')
            else:
                logger.info(f'Finished running {file}. Successful! {len(futures) - i} left to go')
