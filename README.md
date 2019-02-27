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


## Acknowledgments

The research reported in this repository has been supported by the German Research Foundation DFG, as part of Collaborative Research Center (Sonderforschungsbereich) 1320 "EASE – Everyday Activity Science and Engineering", University of Bremen (https://www.ease-crc.org/). The research was conducted in subproject P01 "Embodied semantics for the language of action and change: Combining analysis, reasoning and simulation".
