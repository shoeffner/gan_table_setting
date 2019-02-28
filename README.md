# Learning and generating table settings


## Running training

This project uses [sacred](https://github.com/IDSIA/sacred), so running can be done by using the "usual" sacred commands.

```bash
python tablesetting.py print_config  # To see what can be configured
python tablesetting.py  # Just train
```


### Persisting run results using Sacred, Mongo, and Docker

If you have docker and docker-compose, you can spin up MongoDB and Omniboard instances using the `docker-compose.yml` file.
You can then train and store the results to the MongoDB, and inspect the results at your [local omniboard (http://localhost:9000)](http://localhost:9000).

```bash
docker-compose up
# in new terminal
python tablesetting.py -m sacred
```


### Hyperparameter search

Run the Mongo DB as shown above.
Then use `runner.sh` to run a hyperopt-mongo-worker which can optimize the parameters.
To queue different parameter settings, run:

```bash
python optimize_hyperparameters.py
```

To visualize the results, the `result_server.py` can be used ([http://localhost:5000](http://localhost:5000)).
On startup and every minute, it processes new results coming in from runners.
However, it only forces updates of the most recent result set.
For older result sets (or concurrently running result sets), manual calls are needed:

```bash
python visualize_optimizations.py KEY
```

Where `KEY` is one of the keys stored in `results/runs.list`.


## Acknowledgments

The research reported in this repository has been supported by the German Research Foundation DFG, as part of Collaborative Research Center (Sonderforschungsbereich) 1320 "EASE – Everyday Activity Science and Engineering", University of Bremen (https://www.ease-crc.org/). The research was conducted in subproject P01 "Embodied semantics for the language of action and change: Combining analysis, reasoning and simulation".
