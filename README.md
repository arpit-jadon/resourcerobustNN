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

## Create an ssh session to a running job
Useful commands

### 1. Go to job location
```
condor_ssh_to_job <job_id>
...
logout
```
### 2. Fetch files from job location
```
condor_ssh_to_job -ssh sftp <job_id>
...
sftp> get outputfile.dat
```


## RoCL    

### Evaluation
1. Run the first step of the total_process bash script while passing the trained model for linear classifier training.
2. Previous step gives two models. Now, run the next two steps for robustness evaluation. 
3. Previous step outputs robustness evaluation performance for two epsilon values.


## HYDRA    

There three parts (steps) in HYDRA method. We have to uncomment and run each part sequentially (due to condor).

### Executing code
1. Place the command in `executable_cmd.sh`
2. Update `docker.sub` according to the requirements.
3. Run: `condor_submit docker.sub` (include `-i` flag for interactive mode) 

### Using RoCL adversarial trained weights
1. Include flags `--load_RoCL` with option of parameter `complete` to use with linear layer and parameter `extractor` to use without linear layer.
2. Specify the checkpoint path, e.g. `--source-net ./trained_models/rocl_ckpt_same_attack`.
3. Execute code.

### Black box attack
1. Specify the checkpoint path, e.g. ` --source-net ./trained_models/rocl_ext_adv_base/finetune/latest_exp/checkpoint/checkpoint.pth.tar`.
2. Use the flag ` --black_box_eval`.
3. Execute code/experiment.

## Transfer attack
We create adversarial examples with PGD white-box attack with ResNet-20 auxiliar model. This attack will be considered as black-box attack on HYDRA models.

Create test data using below:
```
python pgd_transfer_attack.py
```

## Experiment results
You can find all results logs in `exp/` directory.