FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

WORKDIR /portfolio-opt

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    openssh-server \
    vim \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir /var/run/sshd && \
    echo 'root:runpod' | chpasswd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /workspace/outputs

ENV output_dir=/workspace/outputs
ENV dataset_dir=/portfolio-opt/jepa/data/parquet_data
ENV PYTHONPATH=/portfolio-opt

ENTRYPOINT ["/bin/bash", "/portfolio-opt/entrypoint.sh"]
