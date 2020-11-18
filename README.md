# Cortx Take Home Problem
Cortx Take Home Problem: Created a deep learning model that can be trained and run with under 16GB GPU using HuggingFace’s Transformers Library that achieves a high result on Google’s Open Domain Question Answering Long Answer task.

# Instructions to install and run code
1. Create an AWS account.
2. Under the All Services Section select EC2 under the compute section
3. Click Launch Instance
    a. Type Deep Learning AMI (Ubuntu 18.04) and select it \\
    b. Then select t2.micro for setting up (change to g4dn.xlarge or p2.xlarge at a later time)
    c. Add 150 GiB of storage
    d. Create a .pem key and launch instance
4. In the Elastic Block Store section: Click Volumes and Create a SSD volume with 100 GiB of memory. Then click actions and attach to same volume. 
5. Use Instance: 
    a. SSH using the following line: ssh -i *.pem -L 8000:localhost:8888 ubuntu@ec2-*.us-east-*.compute.amazonaws.com (* fill based on your instance)
    b. Add SWAP memory from the new volume attached.
        i. Type lsblk into the terminal
        ii. sudo mkswap /dev/* (*volume name)
        iii. sudo swapon /dev/* (*volume name)
6. Clone the following repository
  a. git clone https://github.com/sapereir/CortxHomeProblem.git
  b. Assuming you have setup type: jupyter notebook into terminal 
  c. Then run the entire jupyter notebook

# F1 Score: Precision and Recall

# Instructions to replicate these Results

# Writeup
Please take a look at the repo and read writeup.pdf

# Opinion and Suggestions

# Favorite Charity
