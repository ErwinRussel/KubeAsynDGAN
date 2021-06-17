import torch
import torch.nn as nn
import torch.nn.functional as F
import subprocess, os
from train_generator import TrainGenerator
import re
import threading
import time
from redisai import Client
my_env = os.environ.copy()
my_env["KUBECONFIG"] = os.path.expanduser(f"~/.kube/config")
main_epochs = 100
# TODO: set these epochs
disc_epochs = 100
gen_epochs = 100
refresh_dataset=True

def port_forward(pod):
    subprocess.run("kubectl -n kubeml port-forward " + redispod + " 6379:6379", env=my_env, shell=True)

# port forward to redisAI
out = subprocess.check_output("kubectl -n kubeml get pods", env=my_env, shell=True)
redispod = re.search('redis-([^\s]+)', out.decode("utf-8")).group(0)

print("=====================================")
print("===== ASynDGAN TRAINING PROCESS =====")
print("=====================================")
print("- Port forwarding to pod" + redispod)
th = threading.Thread(target=port_forward, args=(redispod,))
th.start()
print("- Sleep 5 seconds to get the stuff running")
time.sleep(5)

rai = Client(host='localhost', port=6379)

print("--> Initializing and setting discriminator tensor keys")
class Discriminator(nn.Module):
    """ Discriminator
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features//2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features//2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)

    def forward(self, x):
        x.view(x.shape[0], -1)
        y = F.leaky_relu(self.fc1(x), 0.2)
        y = F.dropout(y, 0.3)
        y = F.leaky_relu(self.fc2(y), 0.2)
        y = F.dropout(y, 0.3)
        y = F.leaky_relu(self.fc3(y), 0.2)
        y = F.dropout(y, 0.3)
        y = torch.sigmoid(self.fc4(y))
        return y

discriminator = Discriminator()

print("- Network initialized")
with torch.no_grad():
        for name, layer in discriminator.state_dict().items():
            # Save the weights
            weight_key = f'latest_disc:{name}' 
            rai.tensorset(weight_key, layer.cpu().detach().numpy(), dtype='float32')
print("- Keys set")

for i in range(main_epochs):
    print("===== EPOCH " + str(i + 1) + " =====")
    # # Set Discriminator tensor keys

    # # Generate Dataset
    print("--> Generating new dataset from updated generator")
    out = subprocess.check_output("python3 generate_dataset_train.py", shell=True)
    # print(out)
    print("Dataset Done")

    # Enlist dataset KubeML
    if refresh_dataset:
        print("--> Removing old dataset from KubeML")
        out = subprocess.check_output("./kubeml dataset delete --name mnist_gan", shell=True, env=my_env)
        print(out.decode("utf-8"))
        print("--> Enlisting new dataset in KubeML")
        # TODO: set all these variables globally
        out = subprocess.check_output('./kubeml dataset create --name mnist_gan --traindata x_train_disc.npy --trainlabels y_train_disc.npy --testdata x_test_disc.npy --testlabels y_test_disc.npy', shell=True, env=my_env)
        print(out.decode("utf-8"))

    
    # Train Discriminator with KubeML
    print("--> Starting Training Discriminator on KubeML")
    out = subprocess.check_output("./kubeml train --function disc-load --dataset mnist_gan --epochs 1 --lr 0.0002 --batch 64 --parallelism 3 --static", env=my_env, shell=True)
    out = out.decode("utf-8")
    job_id = re.sub(r"\W", "", out)
    if len(job_id) < 10:
        print("job ID: " + job_id)
    else:
        print(job_id)
        print("Could not find job_id")
        continue

#
    # This could go after we have fixed the discriminator update!
    print("--> Checking if KubeML Discriminator training job is finished.", end='', flush=True)
    while True:
        print('.', end='', flush=True)
        time.sleep(5)
        out = subprocess.check_output("./kubeml task list", env=my_env, shell=True).decode("utf-8")
        if job_id not in out:
            print('')
            print("- Job finished")
            break


    print("--> Setting latest discriminator tensor keys")
    state_dict = dict()
    for name in discriminator.state_dict():
        # load each of the layers in the statedict
        weight_key = f'{job_id}:{name}'
        w = rai.tensorget(weight_key)
        state_dict[weight_key[len(job_id) + 1:]] = torch.from_numpy(w)

    discriminator.load_state_dict(state_dict)
    print("- Keys loaded")

    with torch.no_grad():
        for name, layer in discriminator.state_dict().items():
            # Save the weights
            weight_key = f'latest_disc:{name}' 
            rai.tensorset(weight_key, layer.cpu().detach().numpy(), dtype='float32')
    print("- Keys set")

    # Train Generator locally
    print("--> Train Generator Locally with new Discriminator Model")
    generator_trainer = TrainGenerator(job_id=job_id)
    generator_trainer.train()

print("=====================================")
print("=============== DONE ================")
print("=====================================")
print("- Press ctrl + c to stop the program")