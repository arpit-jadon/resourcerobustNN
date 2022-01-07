# Project_MLCysec

## Creating Docker

The docker file and basic requirements files are already present. Update `Dockerfile` and `requirements.txt` according to need.
We need to create and then upload the docker image to docker hub, to be able to use it in HTcondor.

### 1. Create docker image
```
sudo docker build -f Dockerfile -t mlcysec-docker .
```

### 2. Create docker container (optional)
```
sudo docker container run --name test -it mlcysec-docker
```

### 3. Login to hub
```
sudo docker login
```

### 4. Change tag according to name convention of hub
```
sudo docker tag mlcysec-docker nik1806/mlcysec:1.0
```

### 5. Push to hub
```
sudo docker push nik1806/mlcysec:1.0
```




## HYDRA    

### Executing core
1. Place the command in `executable_cmd.sh`
2. Update `docker.sub` according to the requirements.
3. Run: `condor_submit docker.sub` (include `-i` flag for interactive mode) 