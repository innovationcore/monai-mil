FROM projectmonai/monai
RUN pip3 install clearml
ADD clearml.conf /root
