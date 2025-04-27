# MLPragueImitation
Repository for the 'Accelerating AI Through Human Knowledge: Teaching to Imitate Experts and Win on the Race Track' workshop at ML Prague 2025.

## Link to Colab Notebook

The notebook is already available in Google Colab under this [link](https://colab.research.google.com/drive/1rfh32YFculANOYOGzhwugWnqwQVN_giD?usp=sharing).

## Running the Docker Container

The container is available on dockerhub as `tkietreiber/mlprague-imitation` ([Link](https://hub.docker.com/r/tkietreiber/mlprague-imitation)), you can run it with all necessary arguments via
```bash
docker run -itp 8000:8000 -v $(pwd)/recordings:/app/recordings -v $(pwd)/agents:/app/agents tkietreiber/mlprague-imitation
```
 and open `frontend.html` in a browser. (Just make sure you also created an `agents` and `recordings` folder.)

## Building the Docker Container

Download the `Dockerfile` and run
```bash
docker build . -t tkietreiber/mlprague-imitation
```

Create a directories `recordings` and `agents`, e.g. with `mkdir recordings agents`, then you can start the docker container with
```bash
docker run -itp 8000:8000 -v $(pwd)/recordings:/app/recordings -v $(pwd)/agents:/app/agents tkietreiber/mlprague-imitation
```

Finally, open the `frontend.html` in a browser and play using W,A,S,D.

## Running locally

Clone the repository and make sure you have Python 3.12 installed. (We recommend using a conda environment.) Then run

```bash
pip install -e .
```

to install all necessary requirements.
