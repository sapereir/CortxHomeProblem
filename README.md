# Cortx Take Home Problem
Cortx Take Home Problem: Created a deep learning model that can be trained and run with under 16GB GPU using HuggingFace’s Transformers Library that achieves a high result on Google’s Open Domain Question Answering Long Answer task.

# Instructions to install and run code
1. Create an AWS account.
2. Under the All Services Section select EC2 under the compute section
3. Click Launch Instance
    a. Type Deep Learning AMI (Ubuntu 18.04) and select it 
    b. Then select t2.micro for setting up (change to g4dn.xlarge or p2.xlarge at a later time)
    c. Add 150 GiB of storage
    d. Create a .pem key and launch instance
4. In the Elastic Block Store section: Click Volumes and Create a SSD volume with 50 GiB of memory. Then click actions and attach to same volume. 
5. Use Instance: 
    a. SSH using the following line: ssh -i *.pem -L 8000:localhost:8888 ubuntu@ec2-*.us-east-*.compute.amazonaws.com (* fill based on your instance)
    b. Add SWAP memory from the new volume attached.
        i. Type lsblk into the terminal
        ii. sudo mkswap /dev/* (*volume name)
        iii. sudo swapon /dev/* (*volume name)
6. Clone the following repository: git clone https://github.com/sapereir/CortxHomeProblem.git
7. Download the necessary data and model from the following link: https://drive.google.com/drive/folders/17k8dXden_vkVhq5DX__ggZ-FbRpnIQxz?usp=sharing
   Feel free to download the data directly from Google here: https://ai.google.com/research/NaturalQuestions/download; I used the simplified version but either could be used. However, the non-simplified version sometimes has a runtime error. Additionally, feel free to train the model directly instead of using my pre-trained models. You could use the following link to connect ec2-instance with google drive but the data is small enough to do directly: https://zapier.com/apps/amazon-ec2/integrations/google-drive
8. Running the code:
    a. Type jupyter notebook into the terminal
    b. Change necessary cells (commented) or run cells.
    c. Long term training
        i. Type tmux new -s "name" into terminal 
        ii. Type the following command into the terminal: jupyter nbconvert --to script *.ipynb (* file name)
        iii. Type python *.py (* file name)
        iv. Click control-b + d to exit tmux window
        v. To reattch to window type tmux attach -t "name" into terminal

# F1 Score with Precision and Recall

# 1
Train on part of nq-train-00.jsonl.gz and nq-val-00.jsonl.gz:
    F1 Score: 2.0000
    Precision: 2.4000
    Recall: 1.7143

# 2
Train on nq-train-simplified.jsonl.gz and nq-val-simplified.jsonl.gz:
    F1 Score: 1.4667
    Precision: 1.3750
    Recall: 1.5714

# 3
Train on nq-train-00-XX.jsonl.gz and nq-val-00.jsonl.gz:
    F1 Score: 
    Precision: 
    Recall:
    
# Instructions to replicate these Results


# Writeup
Please take a look at the repo and read writeup.pdf

# Opinion and Suggestions
I really enjoyed the assignment because developing sufficient models that extract answers from entire page of content verus paragraphs is relatively a new problem in the field. I haven't dealt with something at this scale that needs to be robust. I learned a lot primarily from reading a lot of tensorflow code that can be translated to pytorch. Additionally, I learned how I effective I must be in having an efficient data pipeline. Laslty, it prompted me to gauge the improtance in distrbuted models. I love how that problem was open-ended and I didn't have to reimplement architechture based on a particular research paper. I don't have to many suggestions as this was open ended problem and that is the best aspect of it.


# Favorite Charity
Link: https://www.10000degrees.org/
10,000 Degrees helps students from low-income backgrounds get to and through college. 
