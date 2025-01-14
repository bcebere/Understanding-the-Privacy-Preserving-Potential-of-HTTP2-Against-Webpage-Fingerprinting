FROM python:3.9

ENV DEBIAN_FRONTEND=noninteractive
COPY requirements.txt /var/requirements.txt

RUN  apt update && \
     apt install -y --no-install-recommends tshark sudo vim net-tools dnsutils iputils-ping tcpdump python3-pip && \
     apt-get clean && \
     rm -rf /var/lib/apt/lists/*

RUN pip install -r /var/requirements.txt
RUN pip install playwright
RUN python -m playwright install-deps
RUN python -m playwright install
