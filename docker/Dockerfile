FROM tensorflow/tensorflow:latest-gpu
RUN pip install matplotlib
RUN pip install scikit-learn
RUN pip install ultralytics
RUN pip install opencv-python-headless
RUN echo 'Hallo aus dem Image'

RUN apt-get update && apt-get install -y openssh-server

RUN mkdir /var/run/sshd

RUN echo "root:password" | chpasswd

RUN echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config

CMD ["/usr/sbin/sshd", "-D"]