FROM projectmonai/monai:0.9.1
RUN pip3 install clearml
RUN pip3 clearml-agent
ADD clearml.conf /root
