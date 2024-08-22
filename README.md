AWS Setup and EC2 Instance Configuration
Create IAM User:

Permissions: Create a new IAM user and attach administrative permissions. This allows the user to manage all AWS resources. For a specific purpose like MLflow, you may create a custom policy with permissions limited to S3 and EC2.
Policy: If creating a custom policy, make sure it includes permissions such as:
s3:PutObject
s3:GetObject
s3:ListBucket
ec2:DescribeInstances
ec2:StartInstances
ec2:StopInstances
Access Key: Generate an access key for CLI communication. Save the access key ID and secret access key securely.
Create S3 Bucket:

Command: Go to the S3 section of the AWS Management Console, create a new bucket, and name it (e.g., mlflow-demo). This bucket will be used to store MLflow artifacts.
Launch EC2 Instance:

Operating System: Select an operating system (e.g., Ubuntu for security and ease of use).
Instance Type: Choose t2.micro for free tier eligibility.
Key Pair: Create a key pair for SSH access. Download and save the .pem file securely; it’s used for connecting to the instance.
Network Settings:

Allow Traffic: Configure security group settings to allow SSH (port 22), HTTP (port 80), and HTTPS (port 443) traffic. This ensures you can connect to the instance and access MLflow.


all cammand are given below 
sudo apt update
sudo apt install python3-pip
sudo apt install pipx
sudo apt ensurepath
pipx install pipenv
export PATH=$PATH:/home/ubuntu/.local/bin
echo "export PATH=$PATH:/home/ubuntu/.local/bin" >> ~/.bashrc
source ~/.bashrc
mkdir mlflow
cd mlflow
pipenv shell
pipenv install setuptools
pipenv install mlflow
pipenv install awscli
pipenv install boto3
aws configure
mlflow server -h 0.0.0.0 --default-artifact-root s3://mlflow-demo
export MLFOW_TRACKING_URI=http://Public-IPv4-DNS>:5000 this id copy from aws and add :5000 on chrome  like below
tracking Url of Mlflow server 
http://ec2-3-25-219-64.ap-southeast-2.compute.amazonaws.com:5000/