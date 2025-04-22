# MLPragueImitation
Repository for the 'Accelerating AI Through Human Knowledge: Teaching to Imitate Experts and Win on the Race Track' workshop at ML Prague 2025.

# Usage

Download the `Dockerfile` and run
```bash
docker build . -t car-racing
```

Create a directories `recordings` and `agents`, e.g. with `mkdir recordings agents`, then you can start the docker container with
```bash
docker run -itp 8000:8000 -v $(pwd)/recordings:/app/recordings -v $(pwd)/agents:/app/agents car-racing
```

Finally, open the `frontend.html` in a browser and play using W,A,S,D.
